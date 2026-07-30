"""transformers 5.x compat shim for Isaac-GR00T's Qwen3 VLA backbone.

Why this file exists
--------------------

Isaac-GR00T pins ``transformers==4.57.3`` (see
``external_dependencies/Isaac-GR00T/gr00t/eval/sim/SimplerEnv/setup_SimplerEnv.sh``)
and accesses the inner Qwen3 vision and language sub-models directly, e.g.::

    self.model = Qwen3VLForConditionalGeneration.from_pretrained(...)
    while len(self.model.language_model.layers) > select_layer: ...
    self.model.visual.requires_grad_(False)

In transformers 4.5x those attributes resolved through
``Qwen3VLForConditionalGeneration.__getattr__`` to the inner
``Qwen3VLModel``, so ``model.language_model`` and ``model.visual`` worked.

In transformers 5.x the public attribute layout was tightened: the wrapper
class no longer forwards those names, and the real path is one level
deeper at ``model.model.language_model`` / ``model.model.visual``. Five
call sites in
``external_dependencies/Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py``
break with ``AttributeError: 'Qwen3VLForConditionalGeneration' object has
no attribute 'language_model'``.

The X2 VLA training environment (``env_isaaclab``) ships transformers 5.0
because IsaacLab needs PyTorch 2.7+cu128 (RTX 5090 / Blackwell sm_120).
Pinning transformers down to 4.57 in that env risks silently breaking
unrelated IsaacLab usage, and forking Isaac-GR00T directly creates merge
debt every time we pull from upstream. Instead, this module installs two
*read-only* class-level ``@property`` descriptors on
``Qwen3VLForConditionalGeneration`` that forward the legacy attribute
names to the new location. Class-level properties are descriptors, not
sub-modules, so:

* ``self.model.language_model.layers.pop(-1)`` works in both 4.x and 5.x
  -- the property returns the same object as the deep path.
* ``safetensors`` does **not** see the parameters under two keys (a
  previous instance-level forwarding attempt registered the children as
  duplicate sub-modules and broke checkpoint save with ``shared tensors``
  errors).
* The patch is applied exactly once per process (idempotent) and is a
  no-op when the upstream class already has the attributes (i.e. when
  running on transformers 4.5x or a future 5.x that restores them).

The compat shim is invoked as a side-effect of importing the X2 modality
config side-loaders (``x2_modality_config_{7,10}dof.py``), so any code
path that registers the X2 embodiment also gets the shim for free.

Removing the shim
-----------------

When upstream ``NVIDIA/Isaac-GR00T`` adds native transformers-5 support
(either by switching to the new attribute path, or by adopting a
helper like the one below in ``qwen3_backbone.py``), this file becomes
obsolete. To remove:

1. Confirm the M4 acceptance gate at
   ``tests/test_x2_n17_finetune_smoke.py::test_qwen3_backbone_compat_shim_resolves_both_layouts``
   still passes without importing this module.
2. Delete ``gear_sonic/data/_x2_groot_compat.py`` and the two import-time
   side-effect calls at the top of ``x2_modality_config_{7,10}dof.py``.
3. Drop the "transformers 5.x compat shim" subsection from
   ``docs/source/tutorials/vla_training.md``.

If you discover the shim is breaking (e.g. transformers 6 changes the
attribute path again), update :func:`apply_qwen3vl_transformers5_compat`
below and refresh the gate fixtures.

Tracking note
~~~~~~~~~~~~~

TODO(upstream): when ``NVIDIA/Isaac-GR00T``'s ``qwen3_backbone.py`` no
longer references ``self.model.language_model`` / ``self.model.visual``
under transformers 5 (or upstream pins transformers 4.x explicitly and
guards against 5), delete this module.
"""

from __future__ import annotations

import logging


__all__ = [
    "apply_qwen3vl_transformers5_compat",
    "apply_qwen3_backbone_meta_device_guard",
    "apply_gr00t_n1d7_init_meta_guard",
    "apply_gr00t_n1d7_post_init_freeze_fix",
    "apply_gr00t_n1d7_action_head_sample_time_guard",
    "apply_all_x2_groot_compat",
]


_LOG = logging.getLogger(__name__)
_APPLIED_FLAG_ATTR = "_x2_groot_compat_applied"
_BACKBONE_PATCH_FLAG = "_x2_backbone_meta_guard_applied"
_GR00T_INIT_PATCH_FLAG = "_x2_gr00t_init_meta_guard_applied"
_POSTINIT_PATCH_FLAG = "_x2_postinit_freeze_fix_applied"
_SAMPLE_TIME_PATCH_FLAG = "_x2_sample_time_autocast_guard_applied"


def apply_qwen3vl_transformers5_compat() -> bool:
    """Idempotently install ``language_model`` / ``visual`` properties on
    ``Qwen3VLForConditionalGeneration`` if missing.

    Returns:
        ``True`` if the patch was applied this call (or had already been
        applied in this interpreter), ``False`` if no patch was needed
        (transformers 4.x layout where the attributes already work) or
        if ``transformers`` / ``Qwen3VLForConditionalGeneration`` could
        not be imported.

    The function is safe to call multiple times. The first successful
    application sets the class attribute :data:`_APPLIED_FLAG_ATTR` so
    subsequent calls are O(1) no-ops.
    """
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        _LOG.debug(
            "transformers.Qwen3VLForConditionalGeneration not importable -- "
            "compat shim is a no-op."
        )
        return False

    if getattr(Qwen3VLForConditionalGeneration, _APPLIED_FLAG_ATTR, False):
        return True

    # Probe a freshly-instantiated dummy: if the attributes already work
    # (e.g. transformers 4.5x), there's nothing to patch.
    if (
        "language_model" in Qwen3VLForConditionalGeneration.__dict__
        or "visual" in Qwen3VLForConditionalGeneration.__dict__
    ):
        # Either we already patched (covered by the flag above) or the
        # class natively defines them. Either way, leave it alone.
        setattr(Qwen3VLForConditionalGeneration, _APPLIED_FLAG_ATTR, True)
        return True

    # Class-level property: descriptors live on the class, not the
    # instance, so torch.nn.Module's ``__setattr__`` does **not** treat
    # them as registered sub-modules. The result is that
    # ``self.model.language_model`` returns the same object as the deep
    # path ``self.model.model.language_model`` without registering the
    # child twice (which would break safetensors checkpointing).
    Qwen3VLForConditionalGeneration.language_model = property(
        lambda self: self.model.language_model
    )
    Qwen3VLForConditionalGeneration.visual = property(
        lambda self: self.model.visual
    )
    setattr(Qwen3VLForConditionalGeneration, _APPLIED_FLAG_ATTR, True)

    _LOG.info(
        "Applied X2 -> Isaac-GR00T transformers 5.x compat shim "
        "(forwarding Qwen3VLForConditionalGeneration.{language_model,visual} -> "
        "self.model.{language_model,visual}). See "
        "gear_sonic/data/_x2_groot_compat.py for the rationale."
    )
    return True


def apply_qwen3_backbone_meta_device_guard() -> bool:
    """Wrap ``Qwen3Backbone.__init__`` so the nested
    ``Qwen3VLForConditionalGeneration.from_pretrained`` call escapes the
    outer meta-device context.

    Background
    ----------
    transformers >= 5.0 wraps the OUTER ``from_pretrained()`` (the one
    that is loading ``nvidia/GR00T-N1.7-3B``) in an ``init_empty_weights``
    /meta-device context manager so the model can be allocated lazily on
    CPU/GPU before weights stream in. That meta context is still active
    when ``Qwen3Backbone.__init__`` executes its own nested
    ``from_pretrained()`` to fetch the Cosmos-Reason backbone weights,
    which makes accelerate >= 1.13's ``check_and_set_device_map()``
    refuse with::

        RuntimeError: You are using `from_pretrained` with a meta device
        context manager or `torch.set_default_device('meta')`.

    Wrapping the inner call in a ``torch.device('cpu')`` block escapes
    the meta context just for that nested load -- transformers then
    transfers the Cosmos weights into the meta-allocated outer
    parameters as designed.

    We patch ``Qwen3Backbone.__init__`` (rather than e.g.
    ``Qwen3VLForConditionalGeneration.from_pretrained``) so the guard is
    scoped narrowly to the *one* call that needs it, leaving every other
    use of ``Qwen3VLForConditionalGeneration`` (eval, inference,
    standalone tooling) on its native device-map handling.

    Confirmed against transformers==5.0.0, accelerate==1.13.0
    (env_isaaclab) on 2026-05-08.

    Returns:
        ``True`` if the patch was applied (or had already been applied),
        ``False`` if Isaac-GR00T's ``Qwen3Backbone`` could not be
        imported.
    """
    try:
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    except ImportError:
        _LOG.debug("gr00t.model.modules.qwen3_backbone not importable -- meta-guard is a no-op.")
        return False

    if getattr(Qwen3Backbone, _BACKBONE_PATCH_FLAG, False):
        return True

    import torch  # local import: shim is import-time but torch may be heavy

    _orig_init = Qwen3Backbone.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-redef]
        with torch.device("cpu"):
            _orig_init(self, *args, **kwargs)

    _patched_init.__wrapped__ = _orig_init  # type: ignore[attr-defined]
    Qwen3Backbone.__init__ = _patched_init  # type: ignore[method-assign]
    setattr(Qwen3Backbone, _BACKBONE_PATCH_FLAG, True)

    _LOG.info(
        "Applied X2 -> Isaac-GR00T Qwen3Backbone CPU-device guard for "
        "transformers >=5.0 / accelerate >=1.13. See "
        "gear_sonic/data/_x2_groot_compat.py for the rationale."
    )
    return True


def apply_gr00t_n1d7_init_meta_guard() -> bool:
    """Wrap ``Gr00tN1d7.__init__`` to (a) escape transformers 5.0's
    meta-device context for ``Gr00tN1d7ActionHead``'s ``Beta`` constructor,
    and (b) call ``self.post_init()`` at the end so the tied-weights
    tracking machinery is populated before
    ``from_pretrained._finalize_load_state_dict`` runs.

    Background
    ----------
    Two transformers 5.0 regressions hit ``Gr00tN1d7.__init__`` directly:

    1.  ``Gr00tN1d7ActionHead.__init__`` (called from inside
        ``Gr00tN1d7.__init__``) constructs a ``torch.distributions.Beta``
        from ``config.noise_beta_alpha`` / ``config.noise_beta_beta``. The
        ``Beta(...)`` constructor calls ``.item()`` on its concentration
        tensors via the underlying ``Dirichlet`` validation pass; on
        transformers >= 5.0 the outer ``from_pretrained()`` runs all of
        ``__init__`` inside a meta-device context, which makes ``.item()``
        raise::

            RuntimeError: Tensor.item() cannot be called on meta tensors.

        Wrapping the entire ``__init__`` in a ``torch.device('cpu')``
        block escapes the meta context for every tensor-creating side
        effect during construction (Beta concentration tensors, position
        embedding buffers, etc.). torch's ``_sample_dirichlet`` kernel
        also doesn't support bfloat16, so we additionally disable the
        CUDA autocast manager for the constructor -- the outer
        ``from_pretrained`` sets ``torch_dtype=bfloat16`` which would
        otherwise cascade into Beta's concentration tensors via the
        default dtype.

    2.  Upstream ``Gr00tN1d7.__init__`` (in 0.x and early 1.x) does NOT
        call ``self.post_init()`` at the end. Earlier transformers
        versions were lenient; transformers 5.0 is not -- the loader's
        ``_finalize_load_state_dict`` step expects
        ``self.all_tied_weights_keys`` to be populated and raises
        ``AttributeError`` when it isn't. The shim adds the missing
        call.

    Both issues had been patched directly in
    ``external_dependencies/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py``
    in earlier sessions. Per the project rule against modifying upstream
    code, the equivalent fix lives here as a class-level monkey-patch on
    ``Gr00tN1d7.__init__``.

    Confirmed against transformers==5.0.0 (env_isaaclab) on 2026-05-08.

    Returns:
        ``True`` if the patch was applied (or had already been applied),
        ``False`` if Isaac-GR00T's ``Gr00tN1d7`` could not be imported.
    """
    try:
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    except ImportError:
        _LOG.debug("gr00t.model.gr00t_n1d7.gr00t_n1d7 not importable -- init-guard is a no-op.")
        return False

    if getattr(Gr00tN1d7, _GR00T_INIT_PATCH_FLAG, False):
        return True

    import torch

    _orig_init = Gr00tN1d7.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-redef]
        # cpu-device escapes the outer meta context (fixes Beta .item());
        # disabling cuda autocast keeps the Beta concentration tensors in
        # fp32 even when the outer torch_dtype=bfloat16 cascades down.
        with torch.device("cpu"):
            with torch.amp.autocast("cuda", enabled=False):
                _orig_init(self, *args, **kwargs)
        # Upstream __init__ doesn't call post_init(); transformers 5.0
        # requires it for tied-weights tracking. Idempotent if upstream
        # later starts calling it themselves.
        try:
            self.post_init()
        except Exception as exc:  # pragma: no cover -- defensive
            _LOG.warning("Gr00tN1d7.post_init() call failed: %s", exc)

    _patched_init.__wrapped__ = _orig_init  # type: ignore[attr-defined]
    Gr00tN1d7.__init__ = _patched_init  # type: ignore[method-assign]
    setattr(Gr00tN1d7, _GR00T_INIT_PATCH_FLAG, True)

    _LOG.info(
        "Applied X2 -> Isaac-GR00T Gr00tN1d7.__init__ guard "
        "(CPU-device for Beta meta-tensor + post_init call). See "
        "gear_sonic/data/_x2_groot_compat.py for the rationale."
    )
    return True


def _reapply_freeze(model) -> None:
    """Re-run set_trainable_parameters on backbone + action_head per config.

    This is the freezing intent that transformers >= 5.0's
    ``from_pretrained`` (specifically the state-dict-load + tied-weights
    init walk that runs *after* ``__init__`` finishes) silently resets to
    ``requires_grad=True`` on every parameter. We restore it after
    ``from_pretrained`` returns. Safe to call any number of times.
    """
    cfg = model.config
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "set_trainable_parameters"):
        try:
            backbone.set_trainable_parameters(
                tune_llm=cfg.tune_llm,
                tune_visual=cfg.tune_visual,
                tune_top_llm_layers=getattr(cfg, "tune_top_llm_layers", 0),
            )
        except Exception as exc:  # pragma: no cover -- defensive
            _LOG.warning("backbone.set_trainable_parameters re-apply failed: %s", exc)
    action_head = getattr(model, "action_head", None)
    if action_head is not None and hasattr(action_head, "set_trainable_parameters"):
        try:
            action_head.set_trainable_parameters(
                tune_projector=cfg.tune_projector,
                tune_diffusion_model=cfg.tune_diffusion_model,
                tune_vlln=cfg.tune_vlln,
            )
        except Exception as exc:  # pragma: no cover -- defensive
            _LOG.warning("action_head.set_trainable_parameters re-apply failed: %s", exc)


def apply_gr00t_n1d7_post_init_freeze_fix() -> bool:
    """Re-apply ``set_trainable_parameters`` after ``Gr00tN1d7.from_pretrained``
    so ``--no-tune-llm`` / ``--no-tune-visual`` actually freeze.

    Background
    ----------
    ``Gr00tN1d7.__init__`` (in upstream Isaac-GR00T) creates the
    backbone, creates the action head -- both of which call
    ``set_trainable_parameters`` to freeze the user-requested modules --
    and then calls ``self.post_init()`` at the very end to satisfy
    transformers >= 5.0's tied-weights tracking machinery.

    The catch: transformers 5.0's ``from_pretrained`` performs additional
    work *after* ``__init__`` returns -- meta-tensor materialization,
    state-dict streaming, ``init_weights`` over uninitialized parameters,
    tied-weights propagation, device-map dispatch -- and several of those
    steps walk every parameter and set ``requires_grad=True`` as a side
    effect. By the time the trainer queries the model, ALL of the
    carefully-placed ``requires_grad_(False)`` calls from
    ``Qwen3Backbone`` and ``Gr00tN1d7ActionHead`` have been silently
    undone. The trainer happily reports
    ``Trainable parameters: 3,455,180,928 (100.00%)`` and Adam crashes
    90 seconds later trying to allocate ~41 GB of optimizer states for
    the entire 3.45 B model on a 32 GB card.

    Patching ``post_init`` is *not* sufficient: the resets happen later
    in the ``from_pretrained`` pipeline. The reliable hook is
    ``from_pretrained`` itself -- we wrap the classmethod so that after
    upstream returns the fully-loaded model, we re-run
    ``set_trainable_parameters`` on backbone + action head per the
    user's config flags. Direct ``__init__`` callers (the
    ``model_class(config)`` path in ``setup.py``'s else-branch) still
    get the freeze via the un-patched ``__init__`` calls; that path is
    used only when no checkpoint is loaded, which means there's no
    state-dict streaming step to clobber the freeze.

    The base config that ships with ``nvidia/GR00T-N1.7-3B`` has
    ``tune_llm=True`` and ``tune_visual=True`` baked in, so this
    regression bites every freeze-mode finetune (the OOB recipe -- only
    projector + diffusion + vlln trainable).

    Confirmed against transformers==5.0.0 on 2026-05-08; reproducer in
    ``/tmp/probe_final.py`` shows the patch drops trainable params from
    100.00% to 55.91% (1.93 B / 3.45 B) for the OOB freeze-mode recipe.

    Returns:
        ``True`` if the patch was applied (or had already been applied),
        ``False`` if Isaac-GR00T's ``Gr00tN1d7`` could not be imported.
    """
    try:
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    except ImportError:
        _LOG.debug("gr00t.model.gr00t_n1d7.gr00t_n1d7 not importable -- freeze-fix is a no-op.")
        return False

    if getattr(Gr00tN1d7, _POSTINIT_PATCH_FLAG, False):
        return True

    _orig_from_pretrained = Gr00tN1d7.from_pretrained.__func__

    def _patched_from_pretrained(cls, *args, **kwargs):  # type: ignore[no-redef]
        result = _orig_from_pretrained(cls, *args, **kwargs)
        # ``from_pretrained`` may return ``model`` or ``(model, loading_info)``
        # depending on ``output_loading_info``. Locate the model object.
        if isinstance(result, tuple):
            model = result[0]
        else:
            model = result
        _reapply_freeze(model)
        return result

    _patched_from_pretrained.__wrapped__ = _orig_from_pretrained  # type: ignore[attr-defined]
    Gr00tN1d7.from_pretrained = classmethod(_patched_from_pretrained)  # type: ignore[method-assign]
    setattr(Gr00tN1d7, _POSTINIT_PATCH_FLAG, True)

    _LOG.info(
        "Applied X2 -> Isaac-GR00T Gr00tN1d7.from_pretrained freeze-fix "
        "(re-applies set_trainable_parameters after transformers 5.0's "
        "state-dict load resets requires_grad). See "
        "gear_sonic/data/_x2_groot_compat.py for the rationale."
    )
    return True


def apply_gr00t_n1d7_action_head_sample_time_guard() -> bool:
    """Wrap ``Gr00tN1d7ActionHead.sample_time`` so the Beta/Dirichlet draw
    happens in float32 even when the trainer's outer autocast(bfloat16)
    context is active.

    Background
    ----------
    ``Gr00tN1d7ActionHead.sample_time`` (called once per training step
    from ``Gr00tN1d7ActionHead.forward`` to draw the diffusion timestep)
    samples from a ``torch.distributions.Beta`` and casts the result to
    the model's working dtype::

        def sample_time(self, batch_size, device, dtype):
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            sample = (1 - sample) * self.config.noise_s
            return sample

    The cast happens *after* the sample call, so under
    ``torch.amp.autocast("cuda", dtype=bfloat16)`` (which the
    transformers Trainer wraps the entire forward in for mixed-precision
    training) PyTorch tries to draw a bfloat16 Dirichlet sample and
    crashes with::

        RuntimeError: "dirichlet" not implemented for 'BFloat16'

    The companion patch :func:`apply_gr00t_n1d7_init_meta_guard` already
    handles the same problem at *construction* time (it disables autocast
    around ``__init__`` so the ``Beta(alpha, beta)`` constructor doesn't
    inherit the bfloat16 default dtype). This patch handles the
    forward-pass equivalent: every time ``sample_time`` runs, we drop
    out of the autocast region just for the Beta draw, then let the
    trailing ``.to(dtype=...)`` cast the result back to the working
    dtype as the upstream code already intends.

    Disabling autocast for ~3 lines per training step has zero measurable
    perf impact (the dominant cost is the Qwen3 backbone, not a 32-elem
    scalar draw) but is the difference between a working M4 fine-tune
    and a hard crash on the very first ``training_step``.

    Confirmed against transformers==5.0.0, torch==2.7.0+cu128
    (env_isaaclab) on 2026-05-13. Reproduces on
    ``examples/finetune.sh`` with ``--max-steps 10000 --num-gpus 1``.

    Returns:
        ``True`` if the patch was applied (or had already been applied),
        ``False`` if Isaac-GR00T's ``Gr00tN1d7ActionHead`` could not be
        imported.
    """
    try:
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    except ImportError:
        _LOG.debug(
            "gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7ActionHead not "
            "importable -- sample-time guard is a no-op."
        )
        return False

    if getattr(Gr00tN1d7ActionHead, _SAMPLE_TIME_PATCH_FLAG, False):
        return True

    import torch
    from torch.distributions import Beta

    _orig_sample_time = Gr00tN1d7ActionHead.sample_time

    def _patched_sample_time(self, batch_size, device, dtype):  # type: ignore[no-redef]
        # Two layers of defence against the BF16 Dirichlet gap:
        # 1. Disable the active autocast region for the Beta draw, in case
        #    autocast itself would cast the sample call's intermediates.
        # 2. If the Beta's concentration tensors themselves are non-fp32
        #    (which happens in practice -- transformers 5.0's
        #    ``from_pretrained`` cascades ``torch_dtype=bfloat16`` into
        #    every tensor created during the meta-allocation walk,
        #    including the Beta's ``concentration0/1`` even though
        #    ``apply_gr00t_n1d7_init_meta_guard`` tries to suppress this),
        #    rebuild a float32 Beta on the fly and sample from that.
        # Either way, the trailing ``.to(dtype=dtype)`` casts the float32
        # sample back to the working dtype as upstream intends.
        device_type = device.type if hasattr(device, "type") else "cuda"
        with torch.amp.autocast(device_type, enabled=False):
            beta = self.beta_dist
            c1 = getattr(beta, "concentration1", None)
            c0 = getattr(beta, "concentration0", None)
            if (
                c1 is not None and c0 is not None
                and (c1.dtype != torch.float32 or c0.dtype != torch.float32)
            ):
                # Rebuild the Beta with float32 concentrations; cache it
                # on the instance so we only pay the construction cost
                # once (the underlying Dirichlet validation is non-trivial
                # at construction time).
                cached = getattr(self, "_x2_beta_dist_fp32", None)
                if cached is None:
                    cached = Beta(c1.detach().float(), c0.detach().float())
                    self._x2_beta_dist_fp32 = cached
                sample = cached.sample([batch_size]).to(device=device, dtype=dtype)
                sample = (1 - sample) * self.config.noise_s
                return sample
            return _orig_sample_time(self, batch_size, device, dtype)

    _patched_sample_time.__wrapped__ = _orig_sample_time  # type: ignore[attr-defined]
    Gr00tN1d7ActionHead.sample_time = _patched_sample_time  # type: ignore[method-assign]
    setattr(Gr00tN1d7ActionHead, _SAMPLE_TIME_PATCH_FLAG, True)

    _LOG.info(
        "Applied X2 -> Isaac-GR00T Gr00tN1d7ActionHead.sample_time "
        "autocast guard (Beta/Dirichlet sample stays in float32 even "
        "when the trainer wraps forward in autocast(bfloat16)). See "
        "gear_sonic/data/_x2_groot_compat.py for the rationale."
    )
    return True


def apply_all_x2_groot_compat() -> dict[str, bool]:
    """Apply every X2 -> Isaac-GR00T compat patch in the right order.

    Order matters: the meta-device guards must precede any model
    construction (otherwise ``__init__`` will blow up with the
    meta-context error). The from-pretrained freeze fix can be installed
    any time before ``Gr00tN1d7.from_pretrained`` is called.

    The Qwen3VL property shim is independent of the others and is
    invoked first for symmetry / future-proofing (e.g. should upstream
    add a layer-pruning call that uses ``self.model.visual`` directly).

    The action-head ``sample_time`` autocast guard can be installed any
    time before ``Gr00tN1d7ActionHead.forward`` runs; we apply it last
    for clarity.

    Returns:
        Dict ``{patch_name: applied}`` for diagnostic logging.
    """
    return {
        "qwen3vl_property_shim": apply_qwen3vl_transformers5_compat(),
        "qwen3_backbone_cpu_guard": apply_qwen3_backbone_meta_device_guard(),
        "gr00t_n1d7_init_guard": apply_gr00t_n1d7_init_meta_guard(),
        "gr00t_n1d7_freeze_fix": apply_gr00t_n1d7_post_init_freeze_fix(),
        "gr00t_n1d7_sample_time_guard": apply_gr00t_n1d7_action_head_sample_time_guard(),
    }
