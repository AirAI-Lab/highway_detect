import os
import argparse


def list_stems(folder: str, ext: str) -> list:
    stems = []
    for name in os.listdir(folder):
        if name.lower().endswith(ext.lower()):
            stems.append(os.path.splitext(name)[0])
    return sorted(stems)


def write_list(path: str, items: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in items:
            f.write(s + '\n')


def main():
    ap = argparse.ArgumentParser(description='Generate Massachusetts tiff train/val/test split files (stems)')
    ap.add_argument('--root', required=True, help='Path to Massachusetts dataset root containing tiff/ subfolder')
    args = ap.parse_args()

    tiff_root = os.path.join(args.root, 'tiff')
    if not os.path.isdir(tiff_root):
        raise SystemExit(f"Not found: {tiff_root}. Expected <root>/tiff with train/val/test folders.")

    splits = {}
    for subset in ['train', 'val', 'test']:
        img_dir = os.path.join(tiff_root, subset)
        if not os.path.isdir(img_dir):
            print(f"Warning: missing folder {img_dir}, skipping subset '{subset}'")
            continue
        stems = list_stems(img_dir, '.tiff')
        splits[subset] = stems

    out_dir = os.path.join(args.root, 'splits')
    for k, v in splits.items():
        out_path = os.path.join(out_dir, f'{k}.txt')
        write_list(out_path, v)
        print(f"Wrote {len(v)} items to {out_path}")


if __name__ == '__main__':
    main()
