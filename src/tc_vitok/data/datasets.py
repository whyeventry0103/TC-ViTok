"""Dataset loaders for TC-ViTok training and evaluation.

Provides PyTorch Dataset implementations for:
- VQAv2 (primary benchmark)
- GQA (spatial annotation support)
- TextVQA (OCR stress test)
- POPE (hallucination evaluation)
- DummyVQADataset (for pipeline testing without real data)

Paper reference: Section 8.1 (Datasets)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# DINOv2 preprocessing (ImageNet normalization)
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((98, 98)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# CLIP ViT-L/14 preprocessing (336px for LLaVA-1.5)
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


class DummyVQADataset(Dataset):
    """Synthetic dataset for pipeline testing.

    Generates random images and dummy questions/answers.
    Use this to verify the full training pipeline works
    end-to-end before plugging in real data.

    Args:
        num_samples: Number of synthetic samples
        stub_mode: When True, emit pre-computed v_coarse/q_embed features
            instead of raw images_coarse. Use with --stub so the model can
            take the cached forward path without needing real encoders.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        stub_mode: bool = False,
    ):
        self.num_samples = num_samples
        self.stub_mode = stub_mode

        # Pre-generate dummy questions
        self.questions = [
            f"What is in the image? (sample {i})"
            for i in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.stub_mode:
            # Return pre-computed feature tensors so forward_cached() is used,
            # avoiding calls to None encoders when load_encoders=False.
            # v_full (576, 1024) matches CLIP ViT-L/14-336 output shape.
            return {
                "v_coarse": torch.randn(49, 384),
                "q_embed": torch.randn(512),
                "v_full": torch.randn(576, 1024),
                "images_full": torch.randn(3, 336, 336),
                "questions": self.questions[idx],
                "answer_str": "dummy",
            }
        return {
            "images_coarse": torch.randn(3, 98, 98),
            "images_full": torch.randn(3, 336, 336),
            "questions": self.questions[idx],
            "answer_str": "dummy",
        }


class VQAv2Dataset(Dataset):
    """VQAv2 dataset loader for training and evaluation.

    VQAv2 is the primary benchmark. Each sample has an image, a question,
    and multiple annotator answers (10 per question).

    Expected file structure:
        image_dir/COCO_train2014_000000XXXXXX.jpg  (or val2014)
        annotations_path: v2_mscoco_train2014_annotations.json
        questions_path: v2_OpenEnded_mscoco_train2014_questions.json

    Args:
        annotations_path: Path to VQAv2 annotations JSON
        questions_path: Path to VQAv2 questions JSON
        image_dir: Directory containing COCO images
        split: Dataset split ('train' or 'val')
        max_samples: Limit number of samples (for debugging/Phase 0)
    """

    def __init__(
        self,
        annotations_path: str,
        questions_path: str,
        image_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.split = split

        # Load questions
        with open(questions_path) as f:
            questions_data = json.load(f)
        questions_by_id = {
            q["question_id"]: q for q in questions_data["questions"]
        }

        # Load annotations
        with open(annotations_path) as f:
            annotations_data = json.load(f)

        # Build samples list
        self.samples: List[Dict[str, Any]] = []
        for ann in annotations_data["annotations"]:
            qid = ann["question_id"]
            q = questions_by_id.get(qid)
            if q is None:
                continue
            answers = [a["answer"] for a in ann["answers"]]
            # Most frequent answer as the training target
            most_common = max(set(answers), key=answers.count)
            self.samples.append({
                "question_id": qid,
                "image_id": ann["image_id"],
                "question": q["question"],
                "answers": answers,
                "answer": most_common,
            })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Retry on corrupt/missing images — DataLoader worker crashes cause SIGABRT
        for attempt in range(5):
            try:
                return self._load_item((idx + attempt) % len(self.samples))
            except Exception:
                continue
        raise RuntimeError(f"VQAv2Dataset: failed to load any sample near idx={idx}")

    def _load_item(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_id = sample["image_id"]

        # Load image (COCO naming convention)
        img_filename = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        img_path = self.image_dir / img_filename
        image = Image.open(img_path).convert("RGB")

        # Apply both transforms
        images_coarse = DINO_TRANSFORM(image)
        images_full = CLIP_TRANSFORM(image)

        return {
            "images_coarse": images_coarse,
            "images_full": images_full,
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": sample["answers"],
            "question_id": sample["question_id"],
            "image_id": image_id,
        }


class GQADataset(Dataset):
    """GQA dataset loader with spatial annotation support.

    GQA provides structured scene graph annotations that enable
    spatial specificity evaluation (paper Section 8.3).

    Expected file structure:
        image_dir/GQAXXXXXX.jpg (or n-digit ID)
        questions_path: train_balanced_questions.json
        scene_graphs_path: (optional) train_sceneGraphs.json

    Args:
        questions_path: Path to GQA questions JSON
        image_dir: Directory containing GQA images
        scene_graphs_path: Optional path to scene graph annotations
        max_samples: Limit number of samples
    """

    def __init__(
        self,
        questions_path: str,
        image_dir: str,
        scene_graphs_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)

        # Load questions (GQA format: {question_id: {question, imageId, answer, ...}})
        with open(questions_path) as f:
            questions_data = json.load(f)

        # Load scene graphs if available (for spatial classification)
        self.scene_graphs: Dict[str, Dict] = {}
        if scene_graphs_path is not None:
            with open(scene_graphs_path) as f:
                self.scene_graphs = json.load(f)

        # Build samples list
        self.samples: List[Dict[str, Any]] = []
        for qid, qdata in questions_data.items():
            self.samples.append({
                "question_id": qid,
                "image_id": qdata["imageId"],
                "question": qdata["question"],
                "answer": qdata["answer"],
                "types": qdata.get("types", {}),
                "semantic": qdata.get("semantic", {}),
            })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt in range(5):
            try:
                return self._load_item((idx + attempt) % len(self.samples))
            except Exception:
                continue
        raise RuntimeError(f"GQADataset: failed to load any sample near idx={idx}")

    def _load_item(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_id = sample["image_id"]

        # Load image (GQA images are named by imageId)
        img_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        images_coarse = DINO_TRANSFORM(image)
        images_full = CLIP_TRANSFORM(image)

        # Build GQA annotation for spatial classification
        gqa_annotation = {
            "semantic": sample.get("semantic", {}),
            "types": sample.get("types", {}),
        }
        # Merge scene graph annotation if available
        if image_id in self.scene_graphs:
            gqa_annotation["scene_graph"] = self.scene_graphs[image_id]

        return {
            "images_coarse": images_coarse,
            "images_full": images_full,
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": [sample["answer"]],
            "question_id": sample["question_id"],
            "image_id": image_id,
            "gqa_annotation": gqa_annotation,
        }


class TextVQADataset(Dataset):
    """TextVQA dataset for OCR-heavy question answering.

    Used in Phase 2 domain transfer. TextVQA questions require reading
    text in images (signs, labels, etc.).

    Args:
        annotations_path: Path to TextVQA annotations JSON
        image_dir: Directory containing TextVQA images
        max_samples: Limit number of samples
    """

    def __init__(
        self,
        annotations_path: str,
        image_dir: str,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)

        with open(annotations_path) as f:
            data = json.load(f)

        # TextVQA format: {"data": [{question, image_id, answers, ...}]}
        entries = data.get("data", data.get("annotations", []))

        self.samples: List[Dict[str, Any]] = []
        for entry in entries:
            answers = entry.get("answers", [])
            if not answers:
                continue
            most_common = max(set(answers), key=answers.count)
            self.samples.append({
                "question_id": entry.get("question_id", ""),
                "image_id": entry.get("image_id", entry.get("image", "")),
                "question": entry["question"],
                "answers": answers,
                "answer": most_common,
            })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt in range(5):
            try:
                return self._load_item((idx + attempt) % len(self.samples))
            except Exception:
                continue
        raise RuntimeError(f"TextVQADataset: failed to load any sample near idx={idx}")

    def _load_item(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_id = sample["image_id"]

        # TextVQA image IDs are Open Images IDs (hex strings). Images are stored as
        # {image_id}.jpg directly in train_images/. Strip any path prefix that may
        # appear in some annotation versions (e.g. "train/XXXXX" → "XXXXX").
        filename = Path(image_id).name
        candidates = [
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{image_id}.png",
            self.image_dir / f"{filename}.jpg",
            self.image_dir / f"{filename}.png",
        ]
        img_path = next((c for c in candidates if c.exists()), None)
        if img_path is None:
            raise FileNotFoundError(
                f"TextVQA image not found for id={image_id} "
                f"(tried {[str(c) for c in candidates]})"
            )
        image = Image.open(img_path).convert("RGB")

        images_coarse = DINO_TRANSFORM(image)
        images_full = CLIP_TRANSFORM(image)

        return {
            "images_coarse": images_coarse,
            "images_full": images_full,
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": sample["answers"],
            "question_id": sample["question_id"],
            "image_id": image_id,
        }


class LLaVAInstructDataset(Dataset):
    """LLaVA visual instruction tuning data (e.g., llava_instruct_150k.json).

    The official LLaVA-1.5 SFT mixture includes ~150K conversational
    instruction-tuning samples grounded on COCO images. Most published
    token-compression methods on LLaVA-1.5-7B include this mixture; training
    on VQAv2+GQA alone makes our raw VQAv2/POPE numbers lag the literature.

    Format (liuhaotian/LLaVA-Instruct-150K):
        [
          {
            "id": "000000033471",
            "image": "coco/train2017/000000033471.jpg",
            "conversations": [
              {"from": "human", "value": "<image>\\nQuestion?"},
              {"from": "gpt",   "value": "Answer."},
              ...
            ]
          },
          ...
        ]

    This loader takes the FIRST human/gpt turn pair from each conversation as
    a single (question, answer) example. Multi-turn conversations are not
    exposed — TC-ViTok's training signal is single-turn VQA.

    Image resolution: the JSON references `coco/train2017/...`. If only
    train2014 is available locally, the loader tries multiple candidate paths
    and skips samples whose image cannot be located (with retry).

    Args:
        json_path: Path to llava_instruct_150k.json (or compatible LLaVA SFT JSON)
        image_dir: Root directory for COCO images. The loader tries
                   {image_dir}/<filename>, {image_dir}/train2017/<filename>,
                   {image_dir}/train2014/<filename>, and the raw path verbatim.
        max_samples: Optional cap on the number of samples.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)

        with open(json_path) as f:
            data = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        for entry in data:
            convs = entry.get("conversations", [])
            if len(convs) < 2:
                continue
            human = next((c for c in convs if c.get("from") == "human"), None)
            gpt = next((c for c in convs if c.get("from") == "gpt"), None)
            if human is None or gpt is None:
                continue

            question = (human.get("value") or "").replace("<image>", "").strip()
            answer = (gpt.get("value") or "").strip()
            if not question or not answer:
                continue

            self.samples.append({
                "image_path": entry.get("image", ""),
                "question": question,
                "answer": answer,
                "id": entry.get("id", ""),
            })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Retry across adjacent samples to absorb missing-image errors gracefully
        # (LLaVA-Instruct references train2017; if user only has train2014, some
        # images won't be found — skip rather than crash).
        for attempt in range(10):
            try:
                return self._load_item((idx + attempt) % len(self.samples))
            except Exception:
                continue
        raise RuntimeError(
            f"LLaVAInstructDataset: failed to load any sample near idx={idx}"
        )

    def _load_item(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        rel_path = sample["image_path"]
        filename = Path(rel_path).name

        candidates = [
            self.image_dir / rel_path,
            self.image_dir / filename,
            self.image_dir / "train2017" / filename,
            self.image_dir / "train2014" / filename,
        ]
        img_path = next((c for c in candidates if c.exists()), None)
        if img_path is None:
            raise FileNotFoundError(
                f"LLaVA-Instruct image not found: {filename} "
                f"(tried {[str(c) for c in candidates]})"
            )

        image = Image.open(img_path).convert("RGB")
        images_coarse = DINO_TRANSFORM(image)
        images_full = CLIP_TRANSFORM(image)

        return {
            "images_coarse": images_coarse,
            "images_full": images_full,
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": [sample["answer"]],
            "question_id": sample["id"] or str(idx),
            "image_id": filename,
        }


class POPEDataset(Dataset):
    """POPE dataset for object hallucination evaluation.

    POPE is a binary yes/no benchmark. Each sample asks whether
    a specific object exists in the image.

    Args:
        pope_path: Path to POPE JSONL file
        image_dir: Directory containing COCO images
        max_samples: Limit number of samples
    """

    def __init__(
        self,
        pope_path: str,
        image_dir: str,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)

        # POPE format: JSONL with {question, answer, image, category}
        self.samples: List[Dict[str, Any]] = []
        with open(pope_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.samples.append({
                    "question": entry.get("question", entry.get("text", "")),
                    "answer": entry.get("answer", entry.get("label", "")).lower(),
                    "image": entry["image"],
                    "category": entry.get("category", ""),
                })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        img_path = self.image_dir / sample["image"]
        image = Image.open(img_path).convert("RGB")

        images_coarse = DINO_TRANSFORM(image)
        images_full = CLIP_TRANSFORM(image)

        return {
            "images_coarse": images_coarse,
            "images_full": images_full,
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": [sample["answer"]],
            "image_id": sample["image"],
        }


class PairedVQADataset(Dataset):
    """Wrapper that provides paired questions per image for diversity loss.

    The diversity loss (paper Eq. for L_div) requires two different questions
    for the same image in each batch.

    Args:
        base_dataset: Underlying VQA dataset with image_id field
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset
        self._build_index()

    def _build_index(self):
        """Build mapping from image_id to sample indices.

        Uses the base dataset's .samples list directly (if available) to
        avoid calling __getitem__ which loads images from disk. Falls back
        to iterating __getitem__ for datasets without a .samples attribute.
        """
        import random
        self.image_to_indices: Dict[Any, List[int]] = {}

        # Fast path: read from .samples metadata (VQAv2Dataset, GQADataset, etc.)
        if hasattr(self.base, "samples"):
            for i, sample in enumerate(self.base.samples):
                img_id = sample.get("image_id", i)
                if img_id not in self.image_to_indices:
                    self.image_to_indices[img_id] = []
                self.image_to_indices[img_id].append(i)
        else:
            # Slow fallback: must call __getitem__ (e.g., DummyVQADataset)
            for i in range(len(self.base)):
                sample = self.base[i]
                img_id = sample.get("image_id", i)
                if img_id not in self.image_to_indices:
                    self.image_to_indices[img_id] = []
                self.image_to_indices[img_id].append(i)

        self._rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        img_id = sample.get("image_id", idx)
        indices = self.image_to_indices.get(img_id, [idx])

        # Find a different question for the same image
        other_indices = [i for i in indices if i != idx]
        if other_indices:
            pair_idx = self._rng.choice(other_indices)
            pair_sample = self.base[pair_idx]
            sample["paired_question"] = pair_sample["questions"]
        else:
            sample["paired_question"] = sample["questions"]

        return sample


class MMBenchDataset(Dataset):
    """MMBench English dev split.

    Accepts either:
      - A .tsv file (HuggingFace OpenCompass format, base64 images)
      - A .parquet file (lmms-lab/MMBench HuggingFace dataset format)

    Multiple-choice format: each sample has 2-4 options labeled A/B/C/D.
    Scored by exact match against the ground-truth letter.
    """

    def __init__(self, tsv_path: str, max_samples: Optional[int] = None):
        import io

        self._io = io
        self.samples: List[Dict[str, Any]] = []

        if tsv_path.endswith(".parquet"):
            self._load_parquet(tsv_path)
        else:
            self._load_tsv(tsv_path)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _build_prompt(self, question: str, hint: str, opts: list) -> str:
        prompt = ""
        if hint:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        for letter, opt in opts:
            prompt += f"{letter}. {opt}\n"
        prompt += "Answer with the option's letter from the given choices directly."
        return prompt

    def _load_tsv(self, path: str) -> None:
        import base64, csv
        self._b64 = base64
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip().upper()
                hint = row.get("hint", "").strip()
                opts = [(l, row.get(l, "").strip()) for l in ("A", "B", "C", "D")
                        if row.get(l, "").strip() and row.get(l, "").strip().lower() != "nan"]
                if not question or not opts or answer not in {l for l, _ in opts}:
                    continue
                self.samples.append({
                    "index": row.get("index", ""),
                    "image_data": ("b64", row.get("image", "")),
                    "question": self._build_prompt(question, hint, opts),
                    "answer": answer,
                    "category": row.get("category", ""),
                    "l2_category": row.get("l2-category", ""),
                })

    def _load_parquet(self, path: str) -> None:
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            question = str(row.get("question", "") or "").strip()
            answer = str(row.get("answer", "") or "").strip().upper()
            hint = str(row.get("hint", "") or "").strip()
            opts = []
            for letter in ("A", "B", "C", "D"):
                val = str(row.get(letter, "") or "").strip()
                if val and val.lower() not in ("nan", "none", ""):
                    opts.append((letter, val))
            if not question or not opts or answer not in {l for l, _ in opts}:
                continue
            # Image stored as dict {"bytes": b"...", "path": ...} in HF parquet
            img_field = row.get("image", None)
            if img_field is None:
                continue
            if isinstance(img_field, dict):
                img_bytes = img_field.get("bytes", b"")
            else:
                img_bytes = bytes(img_field)
            self.samples.append({
                "index": str(row.get("index", "")),
                "image_data": ("bytes", img_bytes),
                "question": self._build_prompt(question, hint, opts),
                "answer": answer,
                "category": str(row.get("category", "") or ""),
                "l2_category": str(row.get("l2-category", "") or ""),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import base64
        sample = self.samples[idx]
        fmt, data = sample["image_data"]
        if fmt == "b64":
            img_bytes = base64.b64decode(data)
        else:
            img_bytes = data
        image = Image.open(self._io.BytesIO(img_bytes)).convert("RGB")

        return {
            "images_coarse": DINO_TRANSFORM(image),
            "images_full": CLIP_TRANSFORM(image),
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": [sample["answer"]],
            "question_id": sample["index"],
            "category": sample["category"],
            "l2_category": sample["l2_category"],
        }


class MMEDataset(Dataset):
    """MME perception+cognition benchmark.

    Supports two layouts:
      - Traditional: subtask subdirectories with .txt question files + images
      - HuggingFace (lmms-lab/MME): data/*.parquet files

    Score per subtask = accuracy + accuracy_plus where accuracy_plus
    requires both questions for the same image to be correct.
    """

    SUBTASKS = (
        "existence", "count", "position", "color",
        "posters", "celebrity", "scene", "landmark", "artwork", "OCR",
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning",
    )

    def __init__(self, root: str, max_samples: Optional[int] = None):
        self.root = Path(root)
        self.samples: List[Dict[str, Any]] = []

        parquet_dir = self.root / "data"
        if parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
            self._load_parquet(parquet_dir)
        else:
            self._load_files(self.root)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_parquet(self, data_dir: Path) -> None:
        import pandas as pd, io as _io
        dfs = [pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))]
        if not dfs:
            return
        df = pd.concat(dfs, ignore_index=True)
        for _, row in df.iterrows():
            subtask = str(row.get("category", "") or "").strip()
            question = str(row.get("question", "") or "").strip()
            answer = str(row.get("answer", "") or "").strip().lower()
            if not subtask or not question or answer not in ("yes", "no"):
                continue
            # Strip question-number suffix to get image_id for accuracy_plus pairing.
            # question_id format: "existence/000001.jpg_0" -> img_id = "existence/000001.jpg"
            qid = str(row.get("question_id", "") or "")
            img_id = qid.rsplit("_", 1)[0] if qid else f"{subtask}_{len(self.samples)}"
            img_field = row.get("image", None)
            if img_field is None:
                continue
            img_bytes = img_field.get("bytes", b"") if isinstance(img_field, dict) else bytes(img_field)
            self.samples.append({
                "subtask": subtask,
                "image_bytes": img_bytes,
                "image_id": img_id,
                "question": question,
                "answer": answer,
            })

    def _load_files(self, root: Path) -> None:
        for subtask in self.SUBTASKS:
            sub_dir = root / subtask
            if not sub_dir.exists():
                continue
            for q_file in sorted(sub_dir.glob("**/*.txt")):
                with open(q_file) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) < 2:
                            continue
                        question, answer = parts[0], parts[1].lower()
                        img_path = None
                        for ext in (".jpg", ".png", ".jpeg"):
                            for cand in (q_file.with_suffix(ext),
                                         sub_dir / "images" / (q_file.stem + ext)):
                                if cand.exists():
                                    img_path = cand
                                    break
                            if img_path:
                                break
                        if img_path is None:
                            continue
                        self.samples.append({
                            "subtask": subtask,
                            "image_path": str(img_path),
                            "image_id": str(img_path),
                            "question": question,
                            "answer": answer,
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import io as _io
        sample = self.samples[idx]
        if "image_bytes" in sample:
            image = Image.open(_io.BytesIO(sample["image_bytes"])).convert("RGB")
        else:
            image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "images_coarse": DINO_TRANSFORM(image),
            "images_full": CLIP_TRANSFORM(image),
            "questions": sample["question"],
            "answer_str": sample["answer"],
            "answers": [sample["answer"]],
            "image_id": sample["image_id"],
            "subtask": sample["subtask"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for TC-ViTok batches.

    Handles mixed tensor/string/list fields. Passes through
    metadata fields (answers, image_id, etc.) as lists.
    answer_str is passed as a list of strings for generative LM training.
    """
    result = {
        "images_coarse": torch.stack([b["images_coarse"] for b in batch]),
        "images_full": torch.stack([b["images_full"] for b in batch]),
        "questions": [b["questions"] for b in batch],
    }

    # answer_str: raw answer strings for next-token prediction training
    if "answer_str" in batch[0]:
        result["answer_str"] = [b["answer_str"] for b in batch]

    # # Pass through optional metadata fields as lists
    # for key in ["answers", "question_id", "image_id", "gqa_annotation",
    #             "paired_question"]:
    #     if key in batch[0]:
    #         result[key] = [b[key] for b in batch]

    # return result
        # Pass through optional metadata fields as lists safely (handles ConcatDataset mixed batches)
    for key in ["answers", "question_id", "image_id", "gqa_annotation",
                "paired_question", "category", "l2_category", "subtask"]:
        if any(key in b for b in batch):
            result[key] = [b.get(key) for b in batch]

    return result
