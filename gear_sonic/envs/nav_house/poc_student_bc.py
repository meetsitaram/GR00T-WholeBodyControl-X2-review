#!/usr/bin/env python3
"""G4: student forward + BC gradient sanity on G3's synced samples.

Student = ResNet18 (ImageNet init) on 96x96 RGB + MLP on [proprio(6)+goal(6)]
-> 3 sticks. One BC epoch on the 200-sample npz must drop loss >30%, then
ONNX export must succeed. This validates the training plumbing only —
real training is the N2 DAgger loop.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NPZ = "/tmp/claude-1000/poc_g3_samples.npz"


class StudentNet(nn.Module):
    def __init__(self, state_dim: int, act_dim: int = 3):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        r.fc = nn.Identity()
        self.encoder = r
        self.proj = nn.Linear(512, 128)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(), nn.Linear(128, 128))
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, act_dim), nn.Tanh())

    def forward(self, rgb, state):
        f = self.proj(self.encoder(rgb))
        s = self.state_mlp(state)
        return self.head(torch.cat([f, s], dim=1))


def main() -> int:
    d = np.load(NPZ)
    rgb = torch.tensor(d["rgb"]).float().permute(0, 3, 1, 2) / 255.0
    rgb = F.interpolate(rgb, size=(96, 96), mode="bilinear")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = (rgb - mean) / std
    # student state = the NON-privileged slice of the teacher obs:
    # goal-body(2)+dist(1)+goal-yaw sc(2)+has(1) + vel(3) + prev_act(3) = 12
    # (drop the 16 privileged ESDF rays — that's the whole point)
    state = torch.tensor(d["obs"][:, :12]).float()
    act = torch.tensor(d["act"]).float()
    print(f"[g4] data: rgb={tuple(rgb.shape)} state={tuple(state.shape)} "
          f"act={tuple(act.shape)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = StudentNet(state.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    rgb, state, act = rgb.to(dev), state.to(dev), act.to(dev)

    n = rgb.shape[0]
    losses = []
    t0 = time.time()
    for epoch in range(3):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            pred = net(rgb[idx], state[idx])
            loss = F.mse_loss(pred, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        print(f"[g4] epoch {epoch}: loss {np.mean(losses[-max(1, n//32):]):.4f}",
              flush=True)
    first = np.mean(losses[:3])
    last = np.mean(losses[-3:])
    drop = (first - last) / max(first, 1e-9)
    print(f"[g4] loss first3={first:.4f} last3={last:.4f} drop={drop*100:.0f}%")

    onnx_ok = False
    try:
        net.eval()
        torch.onnx.export(
            net.cpu(), (rgb[:1].cpu(), state[:1].cpu()),
            "/tmp/claude-1000/poc_student.onnx",
            input_names=["rgb", "state"], output_names=["sticks"],
            opset_version=17)
        onnx_ok = True
        print("[g4] ONNX export OK -> /tmp/claude-1000/poc_student.onnx")
    except Exception as e:
        print(f"[g4] ONNX export FAILED: {e}")

    verdict = "PASS" if drop > 0.30 and onnx_ok else "FAIL"
    print(f"[g4] VERDICT: {verdict} — BC drop {drop*100:.0f}% "
          f"(need >30%), onnx={'ok' if onnx_ok else 'fail'}, "
          f"{time.time()-t0:.0f}s total")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
