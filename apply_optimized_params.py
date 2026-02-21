import json
import re
import argparse
import sys
import os

DEFAULT_PARAMS_FILE = 'data/best_params_v4.json'
DEFAULT_KEY = 'best_params'
CONFIG_FILE = 'config.py'


def load_params(filepath, key):
    if not os.path.exists(filepath):
        print(f"❌ Error: Parameters file '{filepath}' not found.")
        sys.exit(1)
    with open(filepath, 'r') as f:
        data = json.load(f)

    if key not in data:
        available = list(data.keys())
        print(f"❌ Key '{key}' not found in {filepath}.")
        print(f"   Available keys: {available}")
        sys.exit(1)

    params = data[key]
    if not isinstance(params, dict):
        print(f"❌ Key '{key}' does not contain a dict (got {type(params).__name__}).")
        sys.exit(1)

    return params


def update_config(config_path, params, dry_run=False, auto_yes=False):
    with open(config_path, 'r') as f:
        content = f.read()

    # ── Isolate only the STRATEGY_PARAMS block ─────────────────────
    # Find the block between "STRATEGY_PARAMS = {" and the closing "}"
    # at the same indentation level (first line starting with "}" after the dict)
    sp_match = re.search(
        r'(STRATEGY_PARAMS\s*=\s*\{)(.*?)(^\})',
        content, flags=re.DOTALL | re.MULTILINE
    )
    if not sp_match:
        print("❌ Could not locate STRATEGY_PARAMS block in config. Aborting.")
        sys.exit(1)

    prefix  = content[:sp_match.start(2)]   # everything before the dict body
    sp_body = sp_match.group(2)              # the dict body
    suffix  = content[sp_match.end(2):]      # everything after the closing }

    new_body = sp_body
    changes  = []
    skipped  = []

    print(f"\n🔍 scanning {config_path} for updates (STRATEGY_PARAMS only)...")

    for key, value in params.items():
        # Format value as Python literal
        if isinstance(value, bool):
            py_val = str(value)
        elif isinstance(value, str):
            py_val = f"'{value}'"
        elif isinstance(value, float):
            py_val = f"{value:.6g}"
        else:
            py_val = str(value)

        pattern = (
            r"(^\s*['\"]" + re.escape(key) + r"['\"]\s*:\s*)"
            r"(.*?)"
            r"(\s*,?\s*(?:#.*)?)$"
        )

        found = [False]

        def replacement(match, py_val=py_val, key=key, found=found):
            found[0] = True
            original_val = match.group(2).strip().rstrip(',')
            if original_val != py_val:
                changes.append((key, original_val, py_val))
                return f"{match.group(1)}{py_val}{match.group(3)}"
            return match.group(0)

        new_body = re.sub(pattern, replacement, new_body, flags=re.MULTILINE)

        if not found[0]:
            skipped.append(key)

    if skipped:
        print(f"\n⚠️  {len(skipped)} params not found in config (new params?): {skipped}")
        print("   These need to be added manually if desired.\n")

    if not changes:
        print("✅ No changes needed. All optimized params already match config.")
        return

    print(f"\n📝 Proposed Changes to {config_path}:")
    print(f"  {'Parameter':<28} | {'Old Value':<22} | {'New Value':<22}")
    print("  " + "-" * 76)
    for key, old, new in changes:
        print(f"  {key:<28} | {old:<22} | {new:<22}")
    print("  " + "-" * 76)
    print(f"  {len(changes)} change(s) pending\n")

    if dry_run:
        print("🚫 Dry Run: No files modified.")
        return

    if auto_yes:
        confirm = 'y'
    else:
        confirm = input("❓ Apply these changes? (y/n): ").strip().lower()

    if confirm == 'y':
        new_content = prefix + new_body + suffix
        with open(config_path, 'w') as f:
            f.write(new_content)
        print(f"✅ {len(changes)} param(s) updated in {config_path}")
    else:
        print("❌ Operation cancelled.")



def main():
    parser = argparse.ArgumentParser(description="Apply optimized parameters to config.py")
    parser.add_argument("--params",  default=DEFAULT_PARAMS_FILE,
                        help=f"JSON file with parameters (default: {DEFAULT_PARAMS_FILE})")
    parser.add_argument("--key",     default=DEFAULT_KEY,
                        help=f"Key inside JSON to read params from (default: '{DEFAULT_KEY}')")
    parser.add_argument("--config",  default=CONFIG_FILE,
                        help=f"Target config file (default: {CONFIG_FILE})")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-confirm changes without prompt")
    parser.add_argument("--preview", action="store_true",
                        help="Preview proposed changes only (dry run)")
    args = parser.parse_args()

    params = load_params(args.params, args.key)
    update_config(args.config, params,
                  dry_run=args.preview,
                  auto_yes=args.yes)


if __name__ == "__main__":
    main()
