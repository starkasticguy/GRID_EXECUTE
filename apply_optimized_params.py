import json
import re
import argparse
import sys
import os

DEFAULT_PARAMS_FILE = 'data/best_params_v4.json'
CONFIG_FILE = 'config.py'

def load_params(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: Parameters file '{filepath}' not found.")
        sys.exit(1)
    with open(filepath, 'r') as f:
        return json.load(f)

def update_config(config_path, params, dry_run=False):
    with open(config_path, 'r') as f:
        content = f.read()

    new_content = content
    changes = []

    print(f"\n🔍 scanning {config_path} for updates...")

    for key, value in params.items():
        # Regex explanation:
        # 1. Look for the key in quotes: 'key' or "key"
        # 2. followed by optional whitespace and colon
        # 3. followed by the value (number, bool, or string)
        # 4. followed by comma
        # We want to capture the value group to replace it.
        
        # Adjust value formatting for Python syntax in file
        if isinstance(value, bool):
            py_val = str(value) # True/False
        elif isinstance(value, str):
            py_val = f"'{value}'"
        else:
            py_val = str(value)

        # Pattern: matches line like:  'kama_period': 10,  # Comment
        # Group 1: key quote
        # Group 2: key
        # Group 3: colon + space
        # Group 4: value
        # Group 5: comma
        # Group 6: trailing (comment etc)
        
        # Robust regex:
        # Match 'key' : value ,
        pattern = r"(^\s*['\"]" + re.escape(key) + r"['\"]\s*:\s*)([^,]+)(,.*$)"
        
        # Logic: Replace Group 2 with new value
        def replacement(match):
            original_val = match.group(2).strip()
            # simple check if changed
            # Note: naive string comparison might fail 1.0 vs 1
            if original_val != py_val:
                changes.append((key, original_val, py_val))
                return f"{match.group(1)}{py_val}{match.group(3)}"
            return match.group(0)

        new_content = re.sub(pattern, replacement, new_content, flags=re.MULTILINE)

    if not changes:
        print("✅ No changes needed. Config matches optimized params.")
        return

    print(f"\n📝 Proposed Changes to {config_path}:")
    print(f"{'Parameter':<25} | {'Old Value':<20} | {'New Value':<20}")
    print("-" * 70)
    for key, old, new in changes:
        print(f"{key:<25} | {old:<20} | {new:<20}")
    print("-" * 70)

    if dry_run:
        print("\n🚫 Dry Run: No files modified.")
    else:
        confirm = input("\n❓ Apply these changes? (y/n): ").lower()
        if confirm == 'y':
            with open(config_path, 'w') as f:
                f.write(new_content)
            print(f"✅ Updated {config_path}")
        else:
            print("❌ Operation cancelled.")

def main():
    parser = argparse.ArgumentParser(description="Apply optimized parameters to config.py")
    parser.add_argument("--params", default=DEFAULT_PARAMS_FILE, help="JSON file with parameters")
    parser.add_argument("--config", default=CONFIG_FILE, help="Target python config file")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm changes")
    parser.add_argument("--preview", action="store_true", help="Preview changes only (Dry Run)")
    args = parser.parse_args()

    params = load_params(args.params)
    
    # If preview is set, force dry run
    dry_run = args.preview
    
    # If yes is set, we skip confirmation IF not dry run
    # Implementation detail in update_config uses input().
    # Let's adjust logic.
    
    if args.yes:
        # Hack input to auto-confirm if not preview
        global input
        input = lambda x: 'y'

    update_config(args.config, params, dry_run=dry_run)

if __name__ == "__main__":
    main()
