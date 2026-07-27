#!/usr/bin/env python3
"""Stage-0 nav teacher: PPO on the real kitchen geometry + waypoint registry.

Semantics per gear_sonic/envs/nav_house/nav_kitchen_v1.yaml:
  actions   virtual gamepad — body-frame (fwd, lateral, yaw-rate) sticks with
            deadband -> 0 and certified-envelope magnitude mapping (0.3-1.0);
            one decision per 200 ms tick (the hold floor); action-rate penalty
            prices switching so longer holds emerge; latency DR 0-500 ms.
  motion    velocity-following unicycle (stage-0 stand-in for intent->kplanner;
            stage-1 swaps in kinematic planner playback, same task scaffolding)
  world     nav_grid.npz from build_walkable_mask.py (walkable + ESDF)
  goals     75% random walkable pairs 1-5 m, 25% waypoint pairs (+jitter);
            goal obs is BODY-FRAME relative only
  rewards   progress + time cost + clearance(0.3 m ramp, contact terminal)
            + reach bonus (+heading when waypoint has yaw) + action-rate
  eval      all ordered waypoint routes, deterministic, every N iters

Run (overnight):
    ./train_nav_overnight.sh      # wraps this with logging
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch

KITCHEN = "/home/stickbot/projects/x2-kitchen-sim"


class NavKitchenEnv:
    """rsl_rl VecEnv-compatible vectorized 2D nav env on the kitchen grid."""

    DT = 0.2                    # decision tick (200 ms hold floor)
    SUBSTEPS = 4                # collision checks at 50 ms
    V_MIN, V_MAX = 0.3, 1.0     # certified envelope (m/s fwd/lat, rad/s yaw)
    DEADBAND = 0.15
    CLEAR_RAMP = 0.30
    N_RAYS = 16                 # privileged ESDF probes (teacher only)

    # v2 realism (2026-07-24, user-directed after live wall-contact study:
    # 11 contact events / 15.8 s scraping on a 46.6 s live route):
    #   footprint  — oriented RECTANGLE with wide shoulders (MJCF default-
    #                pose collision extents: fore-aft [-0.168,+0.135] m,
    #                lateral +-0.346 m); the point-robot check legalized
    #                every scrape the live robot performed
    #   quantized  — sticks execute like the deploy daemon: sign-only
    #                fixed magnitudes (0.3 m/s fwd/back/lat, 0.6 rad/s yaw
    #                = KPLANNER_FIXED_TURN_RAD_S rig setting), so margins
    #                are learned under real actuation coarseness
    #   edge ramp  — clearance penalty measured from the BODY EDGE (min
    #                esdf over footprint samples), weight comparable to
    #                progress when scraping
    #   align      — progress discounted when moving backward/sideways in
    #                OPEN space (turn-then-walk preference); front-blocked
    #                states keep full credit so step-back-then-turn stays
    #                free near walls (user rule)
    QF, QY = 0.30, 0.60         # v2 quantized magnitudes (m/s, rad/s)
    EDGE_RAMP = 0.20            # v2 edge-clearance penalty ramp (m)

    def __init__(self, num_envs, device, seed=0, v2=False):
        self.num_envs = num_envs
        self.device = device
        self.v2 = v2
        if v2:
            self.fp_off = torch.tensor(
                [[0.17, 0.35], [0.17, -0.35], [-0.17, 0.35], [-0.17, -0.35],
                 [0.0, 0.35], [0.0, -0.35], [0.17, 0.0], [-0.17, 0.0]],
                dtype=torch.float32, device=device)
        g = np.load(f"{KITCHEN}/assets/nav_grid.npz")
        self.walk = torch.tensor(g["walkable"], device=device)
        self.esdf = torch.tensor(g["esdf"], device=device)
        self.origin = torch.tensor(g["origin"], dtype=torch.float32,
                                   device=device)
        self.res = float(g["res"])
        wps = json.load(open(f"{KITCHEN}/configs/waypoints.json"))
        self.wp_names = sorted(wps)
        self.wp_xy = torch.tensor([wps[k]["xy"] for k in self.wp_names],
                                  dtype=torch.float32, device=device)
        self.wp_yaw = torch.tensor([wps[k]["yaw"] for k in self.wp_names],
                                   dtype=torch.float32, device=device)
        self.wp_rad = torch.tensor([wps[k]["radius"] for k in self.wp_names],
                                   dtype=torch.float32, device=device)
        free = torch.nonzero(self.walk)
        self.free_xy = free.float() * self.res + self.origin + self.res / 2
        if v2:
            # spawn set eroded by the shoulder half-width (+ margin): a
            # random spawn must not start with the footprint in a wall
            e = self.esdf[free[:, 0], free[:, 1]]
            self.spawn_xy = self.free_xy[e >= 0.45]
        else:
            self.spawn_xy = self.free_xy
        # waypoints were labeled with the robot STANDING at counters —
        # often inside the robot-radius erosion band. Snap each to its
        # nearest walkable cell for spawn/eval positions (goal-reach still
        # measures to the labeled point; radius covers the gap).
        d2 = ((self.spawn_xy[None, :, :] - self.wp_xy[:, None, :]) ** 2
              ).sum(-1)
        self.wp_xy_snap = self.spawn_xy[d2.argmin(dim=1)]
        snap_d = (self.wp_xy_snap - self.wp_xy).norm(dim=1)
        # widen reach radius where the labeled point is deep in the band
        self.wp_rad = torch.maximum(self.wp_rad, snap_d + 0.15)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        class _Cfg(dict):          # rsl_rl wandb log_config wants .to_dict()
            def to_dict(self):
                return dict(self)

        self.cfg = _Cfg(
            dt=self.DT, v_range=[self.V_MIN, self.V_MAX],
            deadband=self.DEADBAND, clear_ramp=self.CLEAR_RAMP,
            n_rays=self.N_RAYS, waypoints=self.wp_names,
            num_envs=num_envs, design="nav_kitchen_v1.yaml",
            hardened="latency 0-0.8s, ray noise 0.1, vel noise 0.05, "
                     "heading tol 0.25 (v1 saturated 100% @ iter 400)",
        )

        self.num_actions = 3
        self.num_obs = 2 + 1 + 2 + 1 + 3 + 3 + self.N_RAYS   # 28
        self.max_episode_length = int(60.0 / self.DT)         # 300 ticks
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long,
                                              device=device)
        Z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.pos = Z(num_envs, 2)
        self.yaw = Z(num_envs)
        self.vel_cmd = Z(num_envs, 3)     # commanded (post-envelope) body vel
        self.vel_act = Z(num_envs, 3)     # lagged/actual body vel
        self.goal = Z(num_envs, 2)
        self.goal_yaw = Z(num_envs)
        self.goal_has_yaw = Z(num_envs).bool()
        self.goal_rad = Z(num_envs) + 0.3
        self.prev_dist = Z(num_envs)
        self.prev_act = Z(num_envs, 3)
        self.lag = Z(num_envs)            # latency DR: first-order tau
        self.extras = {"observations": {}}
        ang = torch.arange(self.N_RAYS, device=device) / self.N_RAYS * 2 * math.pi
        self.ray_ang = ang
        self.waypoint_eval_mode = False
        self.reset()

    # -- helpers -----------------------------------------------------------
    def _cell(self, xy):
        c = ((xy - self.origin) / self.res).long()
        c[:, 0].clamp_(0, self.walk.shape[0] - 1)
        c[:, 1].clamp_(0, self.walk.shape[1] - 1)
        return c

    def _esdf_at(self, xy):
        c = self._cell(xy)
        return self.esdf[c[:, 0], c[:, 1]]

    def _walk_at(self, xy):
        c = self._cell(xy)
        return self.walk[c[:, 0], c[:, 1]]

    def _sample_free(self, n):
        i = torch.randint(0, self.spawn_xy.shape[0], (n,), generator=self.rng,
                          device=self.device)
        return self.spawn_xy[i]

    def _reset_idx(self, idx):
        n = int(idx.sum())
        if n == 0:
            return
        use_wp = torch.rand(n, generator=self.rng, device=self.device) < 0.25
        spawn = self._sample_free(n)
        goal = self._sample_free(n)
        gyaw = torch.zeros(n, device=self.device)
        has_yaw = torch.zeros(n, dtype=torch.bool, device=self.device)
        grad = torch.full((n,), 0.3, device=self.device)
        nw = int(use_wp.sum())
        if nw:
            a = torch.randint(0, len(self.wp_names), (nw,), generator=self.rng,
                              device=self.device)
            b = torch.randint(0, len(self.wp_names), (nw,), generator=self.rng,
                              device=self.device)
            b = torch.where(b == a, (b + 1) % len(self.wp_names), b)
            jit = (torch.rand(nw, 2, generator=self.rng, device=self.device)
                   - 0.5) * 0.6
            s = self.wp_xy_snap[a] + jit
            s = torch.where(self._walk_at(s)[:, None], s, self.wp_xy_snap[a])
            spawn[use_wp] = s
            goal[use_wp] = self.wp_xy[b]
            gyaw[use_wp] = self.wp_yaw[b]
            has_yaw[use_wp] = True
            grad[use_wp] = self.wp_rad[b]
        # v3 PIVOT EPISODES (train what the pivot test tests): 15% of
        # episodes the goal IS the spawn position with a random required
        # heading — at-goal 90-180 deg rotations are otherwise a rare
        # state (route arrivals come in pre-aligned via the bearing
        # blend), and the skill collapsed without explicit presentation.
        pivot = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self.v2:
            pivot = torch.rand(n, generator=self.rng,
                               device=self.device) < 0.15
            goal[pivot] = spawn[pivot]
            gyaw[pivot] = (torch.rand(int(pivot.sum()), generator=self.rng,
                                      device=self.device) * 2 - 1) * math.pi
            has_yaw[pivot] = True
            grad[pivot] = 0.4
            if not hasattr(self, "_pivot_ep"):
                self._pivot_ep = torch.zeros(self.num_envs, dtype=torch.bool,
                                             device=self.device)
            self._pivot_ep[idx] = pivot
        # random-goal distance shaping: resample far goals toward 1-5 m
        d = (goal - spawn).norm(dim=1)
        bad = (~use_wp) & (~pivot) & ((d < 1.0) | (d > 5.0))
        for _ in range(4):
            if not bad.any():
                break
            goal[bad] = self._sample_free(int(bad.sum()))
            d = (goal - spawn).norm(dim=1)
            bad = (~use_wp) & (~pivot) & ((d < 1.0) | (d > 5.0))
        self.pos[idx] = spawn
        self.yaw[idx] = (torch.rand(n, generator=self.rng, device=self.device)
                         * 2 - 1) * math.pi
        self.goal[idx] = goal
        self.goal_yaw[idx] = gyaw
        self.goal_has_yaw[idx] = has_yaw
        self.goal_rad[idx] = grad
        self.vel_cmd[idx] = 0
        self.vel_act[idx] = 0
        self.prev_act[idx] = 0
        self.prev_dist[idx] = (goal - spawn).norm(dim=1)
        self.lag[idx] = torch.rand(n, generator=self.rng,
                                   device=self.device) * 0.8
        self.episode_length_buf[idx] = 0
        if self.v2:
            # v3 heading-potential reset: stale cross-episode potentials
            # would pay/charge a bogus first-step heading delta. Same
            # blended target as the reward (bearing far / goal-yaw near).
            if not hasattr(self, "_prev_he"):
                self._prev_he = torch.zeros(self.num_envs, device=self.device)
            d2g = self.goal[idx] - self.pos[idx]
            bear = torch.atan2(d2g[:, 1], d2g[:, 0])
            he_b = torch.atan2(torch.sin(bear - self.yaw[idx]),
                               torch.cos(bear - self.yaw[idx])).abs()
            he_g = torch.atan2(
                torch.sin(self.goal_yaw[idx] - self.yaw[idx]),
                torch.cos(self.goal_yaw[idx] - self.yaw[idx])).abs()
            far = ((d2g.norm(dim=1) - 0.4) / 0.2).clamp(0, 1)
            near_t = torch.where(self.goal_has_yaw[idx], he_g,
                                 torch.zeros_like(he_g))
            self._prev_he[idx] = far * he_b + (1 - far) * near_t

    def _obs(self):
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        d = self.goal - self.pos
        gb = torch.stack((cy * d[:, 0] + sy * d[:, 1],
                          -sy * d[:, 0] + cy * d[:, 1]), dim=1)
        dist = d.norm(dim=1, keepdim=True)
        dyaw = torch.atan2(torch.sin(self.goal_yaw - self.yaw),
                           torch.cos(self.goal_yaw - self.yaw))
        gy = torch.stack((torch.cos(dyaw) * self.goal_has_yaw,
                          torch.sin(dyaw) * self.goal_has_yaw), dim=1)
        # privileged clearance rays (body frame)
        wa = self.yaw[:, None] + self.ray_ang[None, :]
        probes = []
        for r in (0.4, 0.8):
            p = self.pos[:, None, :] + r * torch.stack(
                (torch.cos(wa), torch.sin(wa)), dim=2)
            probes.append(self._esdf_at(p.reshape(-1, 2)).reshape(
                self.num_envs, self.N_RAYS))
        rays = torch.minimum(probes[0], probes[1]).clamp(max=2.0) / 2.0
        if not self.waypoint_eval_mode:      # sensor-noise DR (train only)
            rays = rays + torch.randn(rays.shape, generator=self.rng,
                                      device=self.device) * 0.10
        vel_obs = self.vel_act + (0 if self.waypoint_eval_mode else
                                  torch.randn(self.vel_act.shape,
                                              generator=self.rng,
                                              device=self.device) * 0.05)
        obs = torch.cat([gb / 5.0, dist / 5.0, gy,
                         self.goal_has_yaw[:, None].float(),
                         vel_obs, self.prev_act, rays], dim=1)
        return obs

    def get_observations(self):
        obs = self._obs()
        self.extras["observations"] = {"critic": obs}
        return obs, self.extras

    def reset(self):
        self._reset_idx(torch.ones(self.num_envs, dtype=torch.bool,
                                   device=self.device))
        return self.get_observations()

    def _envelope(self, a):
        """sticks [-1,1]^3 -> certified body velocities with deadband."""
        mag = a.abs().clamp(max=1.0)
        v = torch.where(mag < self.DEADBAND,
                        torch.zeros_like(a),
                        torch.sign(a) * (self.V_MIN + (self.V_MAX - self.V_MIN)
                                         * (mag - self.DEADBAND)
                                         / (1 - self.DEADBAND)))
        return v

    def _fp_points(self, pos, yaw):
        """Footprint sample points (world) for the oriented shoulder-rect."""
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        ox, oy = self.fp_off[:, 0], self.fp_off[:, 1]
        px = pos[:, 0:1] + cy[:, None] * ox[None] - sy[:, None] * oy[None]
        py = pos[:, 1:2] + sy[:, None] * ox[None] + cy[:, None] * oy[None]
        return torch.stack((px, py), dim=2)          # (N, 8, 2)

    def _walk_fp(self, pos, yaw):
        # Footprint vs the TRUE-wall esdf, NOT the walkable mask: walkable
        # is already robot-radius(0.35)-eroded by build_walkable_mask, so
        # checking body-edge samples against it double-erodes (demands
        # 0.70 m lateral — zero valid poses in the kitchen). The oriented
        # rect legally enters the 0.17-0.35 band nose-first, which the v1
        # isotropic disc never could (doorway docking headroom).
        pts = self._fp_points(pos, yaw)
        edge = self._esdf_at(pts.reshape(-1, 2)).reshape(
            pos.shape[0], -1).min(dim=1).values
        return (edge > 0.05) & (self._esdf_at(pos) > 0.10)

    def step(self, actions):
        actions = actions.clamp(-1, 1)
        if self.v2:
            # deploy-quantized execution: sign-only fixed magnitudes
            q = torch.tensor([self.QF, self.QF, self.QY], device=self.device)
            self.vel_cmd = torch.where(actions.abs() < self.DEADBAND,
                                       torch.zeros_like(actions),
                                       torch.sign(actions) * q)
        else:
            self.vel_cmd = self._envelope(actions)
        collided = torch.zeros(self.num_envs, dtype=torch.bool,
                               device=self.device)
        pos0 = self.pos.clone()
        h = self.DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            alpha = 1 - torch.exp(-h / self.lag.clamp(min=1e-3))
            self.vel_act += (self.vel_cmd - self.vel_act) * alpha[:, None]
            cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
            wx = cy * self.vel_act[:, 0] - sy * self.vel_act[:, 1]
            wy = sy * self.vel_act[:, 0] + cy * self.vel_act[:, 1]
            new = self.pos + torch.stack((wx, wy), dim=1) * h
            new_yaw = self.yaw + self.vel_act[:, 2] * h
            ok = (self._walk_fp(new, new_yaw) if self.v2
                  else self._walk_at(new))
            collided |= ~ok
            self.pos = torch.where(ok[:, None], new, self.pos)
            # v2: a blocked footprint also blocks the rotation (a shoulder
            # against the wall cannot sweep through it)
            self.yaw = torch.where(ok, new_yaw, self.yaw) if self.v2 \
                else new_yaw
        self.episode_length_buf += 1

        dist = (self.goal - self.pos).norm(dim=1)
        prog = self.prev_dist - dist
        self.prev_dist = dist
        if self.v2:
            # alignment-discounted progress, gated by front clearance:
            # v3 (user: "it just doesn't want to turn in place") — FULL
            # discount: translation while misaligned in OPEN space earns
            # NOTHING (v2's half-credit still made arcs/strafes beat
            # turn-then-walk); blocked-ahead states keep full credit
            # (step-back-then-turn stays free near walls)
            sp = self.vel_act[:, 0:2].norm(dim=1)
            align = torch.where(sp > 0.05, self.vel_act[:, 0] / (sp + 1e-6),
                                torch.ones_like(sp))
            cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
            ahead = self.pos + 0.45 * torch.stack((cy, sy), dim=1)
            open_sp = ((self._esdf_at(ahead) - 0.45) / 0.30).clamp(0, 1)
            prog = prog * (1.0 - open_sp * (1.0 - align.clamp(min=0)))
            # v3 heading-progress: turning in place toward the RIGHT heading
            # is itself rewarded (potential-based) — the term pivots earn,
            # since prog=0 while rotating. Target BLENDS with distance:
            # far -> bearing-to-goal (corridor turn); inside the arrival
            # zone -> GOAL YAW (align/pivot-at-goal — the user's pivot test:
            # same position, rotated heading, in-place turns only. v1/v2
            # teachers score 0.33/0.11 on it: nothing ever paid for pure
            # pivots). Far component additionally open-space gated; the
            # near component is always on (rotating in place is safe — the
            # footprint check blocks truly jammed rotations).
            d2g = self.goal - self.pos
            bear = torch.atan2(d2g[:, 1], d2g[:, 0])
            he_bear = torch.atan2(torch.sin(bear - self.yaw),
                                  torch.cos(bear - self.yaw)).abs()
            he_goal = torch.atan2(torch.sin(self.goal_yaw - self.yaw),
                                  torch.cos(self.goal_yaw - self.yaw)).abs()
            far = ((dist - 0.4) / 0.2).clamp(0, 1)
            near_t = torch.where(self.goal_has_yaw, he_goal,
                                 torch.zeros_like(he_goal))
            he = far * he_bear + (1 - far) * near_t
            gate = torch.maximum(open_sp * far, 1 - far)
            if not hasattr(self, "_prev_he"):
                self._prev_he = he.clone()
            prog_he = (self._prev_he - he) * gate
            self._prev_he = he.clone()
            # clearance measured from the BODY EDGE (min over footprint).
            # Two ramps (v3, user: "reward being in open spaces, not paths
            # too close to boundaries"): steep don't-scrape inside 0.20 m
            # + a LONG gentle open-space toll from 0.50 m — wall-hugging
            # routes pay it over their whole length, so the corridor
            # centerline wins; a brief doorway pass stays affordable.
            pts = self._fp_points(self.pos, self.yaw)
            clear = self._esdf_at(pts.reshape(-1, 2)).reshape(
                self.num_envs, -1).min(dim=1).values
            ramp = (-1.0 * (self.EDGE_RAMP - clear).clamp(min=0)
                    / self.EDGE_RAMP
                    - 0.15 * (0.50 - clear).clamp(min=0) / 0.50)
        else:
            clear = self._esdf_at(self.pos)
            ramp = -0.5 * (self.CLEAR_RAMP - clear).clamp(min=0) / self.CLEAR_RAMP
            prog_he = torch.zeros_like(prog)
        # v3 maneuver economy (user: time is cheap, wandering is not,
        # pivots are free): movement toll per meter walked — translation
        # pays for itself only when it approaches the goal (net 2.0-0.15
        # per useful meter, strictly negative for wander); in-place turns
        # translate ~nothing so they stay free. Action-switch penalty
        # doubled in v2 so ONE decisive pivot beats many nudges.
        if self.v2:
            walked = (self.pos - pos0).norm(dim=1)
            move_toll = -0.15 * walked
            act_w = 0.10
        else:
            move_toll = 0.0
            act_w = 0.05
        r = (2.0 * prog
             + 0.3 * prog_he
             - 0.01
             + ramp
             + move_toll
             - act_w * (actions - self.prev_act).square().sum(dim=1))
        self.prev_act = actions.clone()

        # pivot episodes ENFORCE in-place (train what the pivot test
        # tests): stepping outside 0.35 m of the spawn/goal during a
        # pivot episode fails like a collision — wander-and-realign paid
        # its way to +10 otherwise (toll -0.15 vs bonus +10) and the
        # learned skill kept collapsing to arcs.
        if self.v2 and hasattr(self, "_pivot_ep"):
            collided |= self._pivot_ep & (dist > 0.35)
        reached = dist < self.goal_rad
        if True:  # heading condition when goal has yaw
            dyaw = torch.atan2(torch.sin(self.goal_yaw - self.yaw),
                               torch.cos(self.goal_yaw - self.yaw)).abs()
            reached &= (~self.goal_has_yaw) | (dyaw < 0.25)
        r += reached.float() * 10.0
        r -= collided.float() * 5.0
        timeout = self.episode_length_buf >= self.max_episode_length
        done = reached | collided | timeout

        # smoothness / hold telemetry: mean per-tick action delta, and mean
        # intent-hold length in ticks (a "change" = any enveloped command
        # component changing by > 0.05). These are what the action-rate
        # penalty is supposed to shape — measure it, don't assume it.
        if not hasattr(self, "_last_raw_act"):
            self._last_raw_act = actions.clone()
            self._hold_ticks = torch.zeros_like(dist)
        delta = (actions - self._last_raw_act)
        changed = delta.abs().max(dim=1).values > 0.05
        self._hold_ticks = torch.where(changed,
                                       torch.zeros_like(self._hold_ticks),
                                       self._hold_ticks + 1)
        self._last_raw_act = actions.clone()
        self.extras["time_outs"] = timeout
        self.extras["log"] = {
            "success_rate": reached.float().mean(),
            "collision_rate": collided.float().mean(),
            "action_delta": delta.norm(dim=1).mean(),
            "hold_len_ticks": self._hold_ticks.mean(),
        }
        if done.any() and not self.waypoint_eval_mode:
            self._reset_idx(done)
        obs = self._obs()
        self.extras["observations"] = {"critic": obs}
        return obs, r, done, self.extras


@torch.no_grad()
def eval_waypoint_routes(env, policy, device):
    """Deterministic run of every ordered waypoint pair. Returns success%."""
    pairs = list(itertools.permutations(range(len(env.wp_names)), 2))
    n = len(pairs)
    ev = NavKitchenEnv(n, device, seed=123, v2=getattr(env, "v2", False))
    ev.waypoint_eval_mode = True
    a = torch.tensor([p[0] for p in pairs], device=device)
    b = torch.tensor([p[1] for p in pairs], device=device)
    ev.pos[:] = ev.wp_xy_snap[a]
    ev.yaw[:] = ev.wp_yaw[a]
    ev.goal[:] = ev.wp_xy[b]
    ev.goal_yaw[:] = ev.wp_yaw[b]
    ev.goal_has_yaw[:] = True
    ev.goal_rad[:] = ev.wp_rad[b]
    ev.vel_cmd[:] = 0; ev.vel_act[:] = 0; ev.prev_act[:] = 0
    ev.prev_dist = (ev.goal - ev.pos).norm(dim=1)
    ev.lag[:] = 0.25
    ev.episode_length_buf[:] = 0
    success = torch.zeros(n, dtype=torch.bool, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    obs = ev._obs()
    for _ in range(ev.max_episode_length):
        act = policy(obs)
        obs, _, done, ex = ev.step(act)
        success |= active & done & ~ex["time_outs"] & (
            (ev.goal - ev.pos).norm(dim=1) < ev.goal_rad + 0.05)
        active &= ~done
        if not active.any():
            break
    return float(success.float().mean()) * 100.0


@torch.no_grad()
def eval_pivot_turns(env, policy, device, disp_tol=0.30):
    """USER CRITERION (2026-07-24): same start/end POSITION, goal heading
    rotated {+90, -90, 180} deg — must pass with in-place turns only
    (net displacement < disp_tol). 3 spawns x 3 rotations = 9 cases."""
    spawns = [torch.tensor([-0.9, -0.3]),
              env.wp_xy_snap[env.wp_names.index("pantry")].cpu(),
              env.wp_xy_snap[env.wp_names.index("hallway")].cpu()]
    rots = [math.pi / 2, -math.pi / 2, math.pi]
    n = len(spawns) * len(rots)
    ev = NavKitchenEnv(n, device, seed=77, v2=getattr(env, "v2", False))
    ev.waypoint_eval_mode = True
    sxy = torch.stack(spawns).float().to(device).repeat_interleave(
        len(rots), dim=0)
    yaw0 = torch.zeros(n, device=device)
    gyaw = yaw0 + torch.tensor(rots * len(spawns), device=device)
    ev.pos[:] = sxy
    ev.yaw[:] = yaw0
    ev.goal[:] = sxy
    ev.goal_yaw[:] = gyaw
    ev.goal_has_yaw[:] = True
    ev.goal_rad[:] = 0.4
    ev.vel_cmd[:] = 0; ev.vel_act[:] = 0; ev.prev_act[:] = 0
    ev.prev_dist = (ev.goal - ev.pos).norm(dim=1)
    ev.lag[:] = 0.25
    ev.episode_length_buf[:] = 0
    if getattr(ev, "v2", False):
        # re-anchor the heading potential to the overridden poses (dist=0
        # here -> blended target = goal yaw)
        ev._prev_he = torch.atan2(torch.sin(ev.goal_yaw - ev.yaw),
                                  torch.cos(ev.goal_yaw - ev.yaw)).abs()
    success = torch.zeros(n, dtype=torch.bool, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    max_disp = torch.zeros(n, device=device)
    obs = ev._obs()
    for _ in range(int(20.0 / ev.DT)):        # 20 s is plenty for 180 deg
        act = policy(obs)
        obs, _, done, ex = ev.step(act)
        max_disp = torch.maximum(max_disp, (ev.pos - sxy).norm(dim=1))
        success |= (active & done & ~ex["time_outs"]
                    & ((ev.goal - ev.pos).norm(dim=1) < ev.goal_rad + 0.05)
                    & (max_disp < disp_tol))
        active &= ~done
        if not active.any():
            break
    out = {}
    labels = ["L90", "R90", "B180"]
    for j, lbl in enumerate(labels):
        m = (torch.arange(n, device=device) % 3) == j
        out[lbl] = round(float(success[m].float().mean()), 3)
    out["max_disp"] = round(float(max_disp.max()), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--run-dir", default=f"{KITCHEN}/runs/nav_teacher")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--v2", action="store_true",
                    help="v2 realism: shoulder-rect footprint, deploy-"
                         "quantized sticks, edge-clearance ramp, "
                         "alignment-discounted progress")
    ap.add_argument("--wandb", action="store_true",
                    help="log to wandb (project x2-kitchen-nav)")
    ap.add_argument("--resume", action="store_true",
                    help="load the latest model_*.pt in run-dir and "
                         "continue training to --iters")
    args = ap.parse_args()
    device = "cuda"
    os.makedirs(args.run_dir, exist_ok=True)

    env = NavKitchenEnv(args.num_envs, device, v2=args.v2)
    env.cfg["v2_realism"] = args.v2
    from rsl_rl.runners import OnPolicyRunner
    if args.wandb:
        # rsl_rl's WandbSummaryWriter forwards step=global_step, but rsl_rl
        # mixes iteration-steps and timestep-steps across metrics; once one
        # large-step metric lands, wandb drops every smaller-step record
        # ("ignoring partial history" — observed emptying the whole
        # dashboard). Patch: no forced step; carry iteration as a field
        # (select it as the x-axis in the wandb UI).
        import wandb
        from torch.utils.tensorboard import SummaryWriter as _TBW
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter as _WSW

        def _add_scalar(self, tag, scalar_value, global_step=None,
                        walltime=None, new_style=False):
            _TBW.add_scalar(self, tag, scalar_value, global_step=global_step,
                            walltime=walltime, new_style=new_style)
            payload = {self._map_path(tag): scalar_value}
            if global_step is not None:
                payload["iteration"] = global_step
            wandb.log(payload)

        _WSW.add_scalar = _add_scalar
    cfg = {
        "num_steps_per_env": 24,
        "save_interval": 500,
        "empirical_normalization": True,
        "logger": "wandb" if args.wandb else "tensorboard",
        "wandb_project": "x2-kitchen-nav",
        "experiment_name": os.path.basename(args.run_dir.rstrip("/")),
        "obs_groups": {"policy": ["critic"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 1.0,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
    runner = OnPolicyRunner(env, cfg, log_dir=args.run_dir, device=device)

    start_iter = 0
    if args.resume:
        import glob as _g
        cks = sorted(_g.glob(f"{args.run_dir}/model_*.pt"),
                     key=lambda p: int(p.split("_")[-1][:-3]))
        if cks:
            runner.load(cks[-1])
            start_iter = int(cks[-1].split("_")[-1][:-3])
            print(f"[nav-teacher] RESUMED from {os.path.basename(cks[-1])} "
                  f"-> continuing to {args.iters}", flush=True)

    t0 = time.time()
    block = args.eval_every
    for start in range(start_iter, args.iters, block):
        runner.learn(num_learning_iterations=block,
                     init_at_random_ep_len=(start == 0))
        policy = runner.get_inference_policy(device=device)
        sr = eval_waypoint_routes(env, policy, device)
        pv = eval_pivot_turns(env, policy, device)
        el = (time.time() - t0) / 3600
        print(f"[nav-teacher] iter {start + block}: waypoint-route success "
              f"{sr:.1f}% pivot={pv} ({el:.2f} h elapsed)", flush=True)
        with open(f"{args.run_dir}/route_eval.jsonl", "a") as fh:
            fh.write(json.dumps({"iter": start + block, "success_pct": sr,
                                 "pivot": pv, "hours": round(el, 3)}) + "\n")
        # route through rsl_rl's writer: a single step authority. A direct
        # wandb.log(step=...) desyncs the counter and wandb then DROPS all
        # of rsl_rl's per-iteration metrics ("ignoring partial history").
        w = getattr(runner, "writer", None)
        if w is not None:
            w.add_scalar("Eval/waypoint_route_success_pct", sr, start + block)
    print("[nav-teacher] training complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
