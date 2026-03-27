"""
YOLO 클래스 선택적 증강 스크립트
────────────────────────────────
대상 클래스 번호와 증강 개수를 직접 지정하여,
해당 클래스가 포함된 이미지만 선택적으로 증강합니다.

사용 예:
    python selective_augment.py \
        --data_dir datasets/my_dataset \
        --target_classes 0 2 \
        --aug_per_image 5 \
        --output_dir datasets/augmented
"""

import argparse
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────
# 1. 증강 파이프라인 정의
# ──────────────────────────────────────────────
def build_pipeline() -> A.Compose:
    """
    Albumentations 증강 파이프라인.
    bbox_params를 YOLO 포맷(yolo)으로 설정하면
    바운딩박스가 자동으로 변환·클리핑됩니다.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(640, 640),
                scale=(0.8, 1.0),
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",          # cx, cy, w, h  (0~1 정규화)
            label_fields=["class_labels"],
            min_visibility=0.3,     # 증강 후 30% 미만 가려진 박스 제거
        ),
    )


# ──────────────────────────────────────────────
# 2. 레이블 파일 읽기 / 쓰기
# ──────────────────────────────────────────────
def read_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO 포맷 레이블 → [(class_id, cx, cy, w, h), ...]
    원본 레이블에 범위(0~1)를 벗어난 좌표가 있을 경우 자동으로 클리핑합니다.
    """
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])

        # ── bbox 정규화: xyxy 공간에서 클리핑 후 xywh 복원 ──
        # 중심점(cx) 이 경계(0 또는 1)에 붙어있으면
        # 기존 min(w, 2*(1-cx)) 공식에서 w=0 이 되므로,
        # 절대좌표(x1,y1,x2,y2)로 변환 후 클리핑하는 방식이 안전합니다.
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        MIN_SIZE = 1e-4  # 정규화 기준으로 너무 작은 박스 제거
        if w < MIN_SIZE or h < MIN_SIZE:
            print(f"[WARN] 유효하지 않은 bbox 스킵: {label_path.name} "
                  f"-> cls={cls} cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
            continue

        rows.append((cls, cx, cy, w, h))
    return rows


def write_label(label_path: Path, rows: list[tuple]) -> None:
    """[(class_id, cx, cy, w, h), ...] → YOLO 포맷 레이블 파일"""
    lines = [f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows]
    label_path.write_text("\n".join(lines))


# ──────────────────────────────────────────────
# 3. 이미지·레이블 경로 수집
# ──────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_samples(data_dir: Path) -> list[tuple[Path, Path]]:
    """
    data_dir 하위에서 이미지-레이블 쌍을 수집합니다.
    images/ ↔ labels/ 폴더 구조 또는 동일 폴더 구조 모두 지원.
    """
    pairs = []
    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        # images/ → labels/ 로 경로 교체 시도 (Windows 백슬래시 대응)
        parts = img_path.parts
        if "images" in parts:
            idx = parts.index("images")
            label_parts = parts[:idx] + ("labels",) + parts[idx + 1:]
            label_path = Path(*label_parts).with_suffix(".txt")
        else:
            label_path = img_path.with_suffix(".txt")

        if not label_path.exists():
            label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


# ──────────────────────────────────────────────
# 4. 핵심 증강 함수
# ──────────────────────────────────────────────
def augment_image(
    img_path: Path,
    label_path: Path,
    target_classes: set[int],
    aug_count: int,
    pipeline: A.Compose,
    out_img_dir: Path,
    out_lbl_dir: Path,
) -> int:
    """
    이미지 1장을 aug_count 회 증강.
    target_classes 중 하나라도 포함된 이미지만 증강합니다.
    반환값: 실제로 저장된 증강 이미지 수
    """
    rows = read_label(label_path)
    if not rows:
        return 0

    # 대상 클래스 포함 여부 확인
    has_target = any(r[0] in target_classes for r in rows)
    if not has_target:
        return 0

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[WARN] 이미지 로드 실패: {img_path}")
        return 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = [(r[1], r[2], r[3], r[4]) for r in rows]
    class_labels = [r[0] for r in rows]

    saved = 0
    stem = img_path.stem
    suffix = img_path.suffix

    for i in range(aug_count):
        try:
            result = pipeline(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
        except Exception as e:
            print(f"[WARN] 증강 실패 ({img_path.name}, {i}번째): {e}")
            continue

        aug_img = result["image"]
        aug_bboxes = result["bboxes"]
        aug_labels = result["class_labels"]

        if len(aug_bboxes) == 0:   # 모든 박스가 사라진 경우 건너뜀
            continue

        out_name = f"{stem}_aug{i:04d}"
        out_img_path = out_img_dir / f"{out_name}{suffix}"
        out_lbl_path = out_lbl_dir / f"{out_name}.txt"

        cv2.imwrite(
            str(out_img_path),
            cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
        )

        new_rows = [
            (int(cls), *bbox) for cls, bbox in zip(aug_labels, aug_bboxes)
        ]
        write_label(out_lbl_path, new_rows)
        saved += 1

    return saved


# ──────────────────────────────────────────────
# 5. 원본 데이터 복사
# ──────────────────────────────────────────────
def copy_originals(
    pairs: list[tuple[Path, Path]],
    out_img_dir: Path,
    out_lbl_dir: Path,
) -> None:
    """원본 이미지·레이블을 출력 폴더에 복사합니다."""
    for img_path, lbl_path in tqdm(pairs, desc="원본 복사"):
        shutil.copy2(img_path, out_img_dir / img_path.name)
        shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)


# ──────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO 클래스 선택적 증강 스크립트"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="원본 데이터셋 루트 경로 (images/, labels/ 하위 구조 또는 혼합)",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        required=True,
        help="증강할 클래스 번호 (복수 지정 가능, 예: --target_classes 0 2)",
    )
    parser.add_argument(
        "--aug_per_image",
        type=int,
        default=3,
        help="대상 이미지 1장당 생성할 증강 이미지 수 (기본값: 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("augmented_dataset"),
        help="증강 결과 출력 경로 (기본값: augmented_dataset/)",
    )
    parser.add_argument(
        "--copy_originals",
        action="store_true",
        default=True,
        help="원본 데이터도 출력 폴더에 복사 (기본값: True)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 랜덤 시드 (기본값: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    target_classes = set(args.target_classes)
    print(f"\n{'='*55}")
    print(f"  대상 클래스  : {sorted(target_classes)}")
    print(f"  이미지당 증강: {args.aug_per_image}장")
    print(f"  입력 경로    : {args.data_dir}")
    print(f"  출력 경로    : {args.output_dir}")
    print(f"{'='*55}\n")

    # 출력 폴더 생성
    out_img_dir = args.output_dir / "images"
    out_lbl_dir = args.output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 수집
    pairs = collect_samples(args.data_dir)
    if not pairs:
        print("[ERROR] 이미지-레이블 쌍을 찾을 수 없습니다. --data_dir 경로를 확인하세요.")
        return
    print(f"총 {len(pairs)}개 이미지-레이블 쌍 발견\n")

    # 원본 복사
    if args.copy_originals:
        copy_originals(pairs, out_img_dir, out_lbl_dir)

    # 증강 파이프라인 빌드
    pipeline = build_pipeline()

    # 증강 실행
    total_saved = 0
    skipped = 0
    for img_path, lbl_path in tqdm(pairs, desc="클래스 선택적 증강"):
        saved = augment_image(
            img_path=img_path,
            label_path=lbl_path,
            target_classes=target_classes,
            aug_count=args.aug_per_image,
            pipeline=pipeline,
            out_img_dir=out_img_dir,
            out_lbl_dir=out_lbl_dir,
        )
        total_saved += saved
        if saved == 0:
            skipped += 1

    print(f"\n{'='*55}")
    print(f"  증강 완료!")
    print(f"  대상 클래스 미포함(스킵): {skipped}장")
    print(f"  생성된 증강 이미지      : {total_saved}장")
    print(f"  최종 데이터셋 크기      : {len(pairs) + total_saved}장")
    print(f"  저장 위치               : {args.output_dir.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
