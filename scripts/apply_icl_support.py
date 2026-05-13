#!/usr/bin/env python3
"""Patch iclpi to consume human-video support context in pi0/pi0.5 training.

Run from the iclpi repository root:

    python scripts/apply_icl_support_patch.py

The script creates .bak_icl backups before modifying files. It is intentionally
conservative: if an expected anchor is missing, it stops with an actionable error.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.cwd()


def path(rel: str) -> Path:
    return ROOT / rel


def read(rel: str) -> str:
    p = path(rel)
    if not p.exists():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    p = path(rel)
    if p.exists():
        bak = p.with_suffix(p.suffix + ".bak_icl")
        if not bak.exists():
            shutil.copy2(p, bak)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"[OK] patched {rel}")


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    if old not in text:
        raise RuntimeError(f"Anchor not found for {name}")
    if new in text:
        print(f"[SKIP] already patched: {name}")
        return text
    return text.replace(old, new, 1)


def regex_replace_once(text: str, pattern: str, repl: str, *, name: str, flags: int = re.DOTALL) -> str:
    if re.search(pattern, text, flags=flags) is None:
        raise RuntimeError(f"Regex anchor not found for {name}")
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"Unexpected replacement count {n} for {name}")
    return new_text


def patch_transforms() -> None:
    rel = "src/openpi/transforms.py"
    text = read(rel)
    if "META_KEYS_TO_PRESERVE" not in text:
        text = text.replace(
            "DataDict: TypeAlias = at.PyTree\n",
            "DataDict: TypeAlias = at.PyTree\n\n"
            "# Metadata needed by in-context support-video loading. These keys are not\n"
            "# normalized and are kept outside the robot observation image dict.\n"
            "META_KEYS_TO_PRESERVE = (\"episode_index\", \"frame_index\", \"task_index\", \"support_round_id\")\n",
            1,
        )
    old = """    def __call__(self, data: DataDict) -> DataDict:\n        flat_item = flatten_dict(data)\n        return jax.tree.map(lambda k: flat_item[k], self.structure)\n"""
    new = """    def __call__(self, data: DataDict) -> DataDict:\n        flat_item = flatten_dict(data)\n        repacked = jax.tree.map(lambda k: flat_item[k], self.structure)\n        # Preserve lightweight metadata for support-context lookup. This avoids\n        # changing every DataConfig repack structure by hand.\n        for key in META_KEYS_TO_PRESERVE:\n            if key in data and key not in repacked:\n                repacked[key] = data[key]\n        return repacked\n"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "repacked = jax.tree.map" not in text:
        raise RuntimeError("Could not patch RepackTransform.__call__; please patch manually.")
    write(rel, text)


def patch_aloha_policy() -> None:
    rel = "src/openpi/policies/aloha_policy.py"
    text = read(rel)
    old = """        if \"prompt\" in data:\n            inputs[\"prompt\"] = data[\"prompt\"]\n        return inputs\n"""
    new = """        if \"prompt\" in data:\n            inputs[\"prompt\"] = data[\"prompt\"]\n\n        # Preserve metadata used by in-context support lookup. These fields are\n        # not robot observations and must not be mixed into the image dict.\n        for key in (\"episode_index\", \"frame_index\", \"task_index\", \"support_round_id\"):\n            if key in data:\n                inputs[key] = data[key]\n\n        return inputs\n"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "support_round_id" not in text:
        raise RuntimeError("Could not patch AlohaInputs metadata preservation.")
    write(rel, text)


def patch_config() -> None:
    rel = "src/openpi/training/config.py"
    text = read(rel)
    if "use_support_context" not in text.split("class GroupFactory", 1)[0]:
        anchor = """    local_files_only: bool = False\n"""
        insert = """    local_files_only: bool = False\n\n    # In-context support-video options. These are no-ops when use_support_context=False.\n    use_support_context: bool = False\n    support_manifest_path: str | None = None\n    support_rounds_per_cycle: int = 1\n    support_cache_size: int = 1024\n    num_support_frames: int = 8\n    support_caption_max_len: int = 128\n"""
        text = replace_once(text, anchor, insert, name="DataConfig support fields")
    write(rel, text)


def patch_data_loader() -> None:
    rel = "src/openpi/training/data_loader.py"
    text = read(rel)
    if "openpi.training.support_context as _support_context" not in text:
        text = text.replace(
            "import openpi.training.config as _config\n",
            "import openpi.training.config as _config\nimport openpi.training.support_context as _support_context\n",
            1,
        )

    old = """    return TransformedDataset(\n        dataset,\n        [\n            *data_config.repack_transforms.inputs,\n            *data_config.data_transforms.inputs,\n            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),\n            *data_config.model_transforms.inputs,\n        ],\n    )\n"""
    new = """    input_transforms = [\n        *data_config.repack_transforms.inputs,\n        *data_config.data_transforms.inputs,\n    ]\n    if data_config.use_support_context:\n        if data_config.support_manifest_path is None:\n            raise ValueError(\"use_support_context=True requires support_manifest_path\")\n        input_transforms.append(\n            _support_context.AddSupportContext(\n                data_config.support_manifest_path,\n                num_support_frames=data_config.num_support_frames,\n                support_cache_size=data_config.support_cache_size,\n                support_caption_max_len=data_config.support_caption_max_len,\n            )\n        )\n    input_transforms.extend([\n        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),\n        *data_config.model_transforms.inputs,\n    ])\n    return TransformedDataset(dataset, input_transforms)\n"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "AddSupportContext" not in text:
        raise RuntimeError("Could not patch transform_dataset support insertion.")

    old2 = """    dataset = create_torch_dataset(data_config, action_horizon, model_config)\n    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)\n"""
    new2 = """    dataset = create_torch_dataset(data_config, action_horizon, model_config)\n    if data_config.use_support_context:\n        dataset = _support_context.SupportRoundDataset(\n            dataset,\n            support_rounds_per_cycle=data_config.support_rounds_per_cycle,\n        )\n    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)\n"""
    if old2 in text:
        text = text.replace(old2, new2, 1)
    elif "SupportRoundDataset" not in text:
        raise RuntimeError("Could not patch create_torch_data_loader support wrapper.")
    write(rel, text)


def patch_model() -> None:
    rel = "src/openpi/models/model.py"
    text = read(rel)
    if "support_images:" not in text:
        anchor = """    token_loss_mask: at.Bool[ArrayT, \"*b l\"] | None = None\n"""
        repl = anchor + """\n    # In-context support-video fields. These are separate from robot observation images.\n    support_images: at.Float[ArrayT, \"*b k h w c\"] | None = None\n    support_image_mask: at.Bool[ArrayT, \"*b k\"] | None = None\n    support_frame_progress: at.Float[ArrayT, \"*b k\"] | None = None\n    chunk_progress: at.Float[ArrayT, \"*b one\"] | None = None\n    support_caption_tokens: at.Int[ArrayT, \"*b l\"] | None = None\n    support_caption_mask: at.Bool[ArrayT, \"*b l\"] | None = None\n"""
        text = replace_once(text, anchor, repl, name="Observation support fields")

    if "support_caption_tokens and support_caption_mask" not in text:
        anchor = """        if (\"tokenized_prompt\" in data) != (\"tokenized_prompt_mask\" in data):\n            raise ValueError(\"tokenized_prompt and tokenized_prompt_mask must be provided together.\")\n"""
        repl = anchor + """        if (\"support_caption_tokens\" in data) != (\"support_caption_mask\" in data):\n            raise ValueError(\"support_caption_tokens and support_caption_mask must be provided together.\")\n"""
        text = replace_once(text, anchor, repl, name="support caption mask validation")

    if "support_images = data.get(\"support_images\")" not in text:
        anchor = """        for key in data[\"image\"]:\n            if data[\"image\"][key].dtype == np.uint8:\n                data[\"image\"][key] = data[\"image\"][key].astype(np.float32) / 255.0 * 2.0 - 1.0\n            elif hasattr(data[\"image\"][key], \"dtype\") and data[\"image\"][key].dtype == torch.uint8:\n                data[\"image\"][key] = data[\"image\"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0\n"""
        repl = anchor + """\n        support_images = data.get(\"support_images\")\n        if support_images is not None:\n            if getattr(support_images, \"dtype\", None) == np.uint8:\n                data[\"support_images\"] = support_images.astype(np.float32) / 255.0 * 2.0 - 1.0\n            elif hasattr(support_images, \"dtype\") and support_images.dtype == torch.uint8:\n                data[\"support_images\"] = support_images.to(torch.float32) / 255.0 * 2.0 - 1.0\n"""
        text = replace_once(text, anchor, repl, name="support image normalization")

    old = """            tokenized_prompt_mask=data.get(\"tokenized_prompt_mask\"),\n            token_ar_mask=data.get(\"token_ar_mask\"),\n            token_loss_mask=data.get(\"token_loss_mask\"),\n        )\n"""
    new = """            tokenized_prompt_mask=data.get(\"tokenized_prompt_mask\"),\n            token_ar_mask=data.get(\"token_ar_mask\"),\n            token_loss_mask=data.get(\"token_loss_mask\"),\n            support_images=data.get(\"support_images\"),\n            support_image_mask=data.get(\"support_image_mask\"),\n            support_frame_progress=data.get(\"support_frame_progress\"),\n            chunk_progress=data.get(\"chunk_progress\"),\n            support_caption_tokens=data.get(\"support_caption_tokens\"),\n            support_caption_mask=data.get(\"support_caption_mask\"),\n        )\n"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "support_caption_mask=data.get" not in text:
        raise RuntimeError("Could not patch Observation.from_dict return.")

    old2 = """        tokenized_prompt_mask=observation.tokenized_prompt_mask,\n        token_ar_mask=observation.token_ar_mask,\n        token_loss_mask=observation.token_loss_mask,\n    )\n"""
    new2 = """        tokenized_prompt_mask=observation.tokenized_prompt_mask,\n        token_ar_mask=observation.token_ar_mask,\n        token_loss_mask=observation.token_loss_mask,\n        support_images=observation.support_images,\n        support_image_mask=observation.support_image_mask,\n        support_frame_progress=observation.support_frame_progress,\n        chunk_progress=observation.chunk_progress,\n        support_caption_tokens=observation.support_caption_tokens,\n        support_caption_mask=observation.support_caption_mask,\n    )\n"""
    if old2 in text:
        text = text.replace(old2, new2, 1)
    elif "support_images=observation.support_images" not in text:
        raise RuntimeError("Could not patch preprocess_observation return.")
    write(rel, text)


def patch_pi0_config() -> None:
    rel = "src/openpi/models/pi0_config.py"
    text = read(rel)
    if "use_support_context" not in text:
        anchor = """    discrete_state_input: bool = None  # type: ignore\n"""
        repl = anchor + """\n    # In-context support-video options. Disabled by default for vanilla pi0/pi0.5.\n    use_support_context: bool = False\n    num_support_frames: int = 8\n    use_support_caption: bool = True\n    use_support_progress: bool = True\n    use_support_role_embedding: bool = True\n    support_caption_max_len: int = 128\n    support_progress_embed_dim: int = 128\n"""
        text = replace_once(text, anchor, repl, name="Pi0Config support fields")

    if "support_kwargs = {}" not in text:
        anchor = """        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)\n        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)\n        with at.disable_typechecking():\n"""
        repl = """        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)\n        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)\n        support_kwargs = {}\n        if self.use_support_context:\n            support_kwargs = {\n                \"support_images\": jax.ShapeDtypeStruct(\n                    [batch_size, self.num_support_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32\n                ),\n                \"support_image_mask\": jax.ShapeDtypeStruct([batch_size, self.num_support_frames], jnp.bool_),\n                \"support_frame_progress\": jax.ShapeDtypeStruct([batch_size, self.num_support_frames], jnp.float32),\n                \"chunk_progress\": jax.ShapeDtypeStruct([batch_size, 1], jnp.float32),\n                \"support_caption_tokens\": jax.ShapeDtypeStruct([batch_size, self.support_caption_max_len], jnp.int32),\n                \"support_caption_mask\": jax.ShapeDtypeStruct([batch_size, self.support_caption_max_len], jnp.bool_),\n            }\n        with at.disable_typechecking():\n"""
        text = replace_once(text, anchor, repl, name="Pi0Config support input specs")

    old = """                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),\n            )\n"""
    new = """                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),\n                **support_kwargs,\n            )\n"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "**support_kwargs" not in text:
        raise RuntimeError("Could not patch Observation spec kwargs.")
    write(rel, text)


def patch_pi0() -> None:
    rel = "src/openpi/models/pi0.py"
    text = read(rel)
    if "self.use_support_context" not in text:
        anchor = """        self.PaliGemma = nnx.Dict(llm=llm, img=img)\n"""
        repl = anchor + """\n        self.use_support_context = config.use_support_context\n        self.use_support_caption = config.use_support_caption\n        self.use_support_progress = config.use_support_progress\n        self.use_support_role_embedding = config.use_support_role_embedding\n        self.support_progress_embed_dim = config.support_progress_embed_dim\n        if self.use_support_context:\n            width = paligemma_config.width\n            if self.use_support_progress:\n                self.support_progress_mlp_in = nnx.Linear(2 * config.support_progress_embed_dim, width, rngs=rngs)\n                self.support_progress_mlp_out = nnx.Linear(width, width, rngs=rngs)\n            if self.use_support_role_embedding:\n                # roles: 0=robot observation image, 1=support image, 2=support caption, 3=task prompt\n                self.prefix_role_embeddings = nnx.Param(jnp.zeros((4, width), dtype=jnp.float32))\n"""
        text = replace_once(text, anchor, repl, name="Pi0 support modules")

    pattern = r"    @at\.typecheck\n    def embed_prefix\(.*?\n    @at\.typecheck\n    def embed_suffix"
    repl = r'''    def _add_prefix_role(self, tokens, role_id: int):
        if self.use_support_context and self.use_support_role_embedding:
            role = self.prefix_role_embeddings.value[role_id].astype(tokens.dtype)
            return tokens + role
        return tokens

    def _support_progress_embedding(self, support_frame_progress, chunk_progress, tokens_per_frame: int):
        batch_size, num_frames = support_frame_progress.shape
        p = support_frame_progress.reshape(-1)
        q = jnp.broadcast_to(chunk_progress, (batch_size, num_frames)).reshape(-1)
        p_emb = posemb_sincos(p, self.support_progress_embed_dim, min_period=4e-3, max_period=4.0)
        q_emb = posemb_sincos(q, self.support_progress_embed_dim, min_period=4e-3, max_period=4.0)
        progress = jnp.concatenate([p_emb, q_emb], axis=-1)
        progress = self.support_progress_mlp_in(progress)
        progress = nnx.swish(progress)
        progress = self.support_progress_mlp_out(progress)
        progress = einops.rearrange(progress, "(b k) d -> b k d", b=batch_size, k=num_frames)
        return einops.repeat(progress, "b k d -> b (k s) d", s=tokens_per_frame)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        # Robot observation images: only the robot's current cameras belong here.
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_tokens = self._add_prefix_role(image_tokens, 0)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # In-context support video frames. Kept separate from robot observation images.
        if self.use_support_context:
            if obs.support_images is None or obs.support_image_mask is None:
                raise ValueError("use_support_context=True requires support_images and support_image_mask")
            b, k = obs.support_images.shape[:2]
            support_images = einops.rearrange(obs.support_images, "b k h w c -> (b k) h w c")
            support_tokens, _ = self.PaliGemma.img(support_images, train=False)
            tokens_per_frame = support_tokens.shape[1]
            support_tokens = einops.rearrange(support_tokens, "(b k) s d -> b (k s) d", b=b, k=k)
            support_tokens = self._add_prefix_role(support_tokens, 1)
            if self.use_support_progress:
                if obs.support_frame_progress is None or obs.chunk_progress is None:
                    raise ValueError("support progress requires support_frame_progress and chunk_progress")
                support_tokens = support_tokens + self._support_progress_embedding(
                    obs.support_frame_progress,
                    obs.chunk_progress,
                    tokens_per_frame,
                ).astype(support_tokens.dtype)
            tokens.append(support_tokens)
            input_mask.append(einops.repeat(obs.support_image_mask, "b k -> b (k s)", s=tokens_per_frame))
            ar_mask += [False] * support_tokens.shape[1]

            # Support caption is a global description of the whole support video.
            if self.use_support_caption:
                if obs.support_caption_tokens is None or obs.support_caption_mask is None:
                    raise ValueError("use_support_caption=True requires support_caption_tokens and support_caption_mask")
                support_caption_tokens = self.PaliGemma.llm(obs.support_caption_tokens, method="embed")
                support_caption_tokens = self._add_prefix_role(support_caption_tokens, 2)
                tokens.append(support_caption_tokens)
                input_mask.append(obs.support_caption_mask)
                ar_mask += [False] * support_caption_tokens.shape[1]

        # Task prompt tokens.
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokenized_inputs = self._add_prefix_role(tokenized_inputs, 3)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix'''
    if "def _support_progress_embedding" not in text:
        text = regex_replace_once(text, pattern, repl, name="Pi0 embed_prefix replacement")
    write(rel, text)


def patch_weight_loaders() -> None:
    rel = "src/openpi/training/weight_loaders.py"
    text = read(rel)
    if "missing_regex:" not in text.split("class PaliGemmaWeightLoader", 1)[0]:
        anchor = """    params_path: str\n"""
        repl = """    params_path: str\n    # Missing params matching this regex are kept from the freshly initialized model.\n    # This allows loading vanilla pi0/pi0.5 checkpoints into an in-context model.\n    missing_regex: str = r\".*(lora.*|support_.*|prefix_role_embeddings.*).*\"\n"""
        text = replace_once(text, anchor, repl, name="CheckpointWeightLoader missing_regex field")
    old = """        return _merge_params(loaded_params, params, missing_regex=\".*lora.*\")\n"""
    new = """        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)\n"""
    if old in text:
        text = text.replace(old, new, 1)
    elif "missing_regex=self.missing_regex" not in text:
        raise RuntimeError("Could not patch CheckpointWeightLoader.load")
    write(rel, text)


def copy_support_context() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "openpi" / "training" / "support_context.py"
    dst = path("src/openpi/training/support_context.py")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        bak = dst.with_suffix(dst.suffix + ".bak_icl")
        if not bak.exists():
            shutil.copy2(dst, bak)
    shutil.copy2(src, dst)
    print("[OK] installed src/openpi/training/support_context.py")


def main() -> None:
    copy_support_context()
    patch_transforms()
    patch_aloha_policy()
    patch_config()
    patch_data_loader()
    patch_model()
    patch_pi0_config()
    patch_pi0()
    patch_weight_loaders()
    print("\n[DONE] ICL support patch applied. Backups use the .bak_icl suffix.")


if __name__ == "__main__":
    main()
