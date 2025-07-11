#!/usr/bin/env python3
"""
ChordiSpeak - AI Chord Vocal Generator
Copyright (C) 2024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Version management script for ChordiSpeak
"""

import re
import sys
from pathlib import Path

def read_version():
    """Read current version from VERSION file"""
    try:
        with open('VERSION', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "1.0.0"

def write_version(version):
    """Write version to VERSION file"""
    with open('VERSION', 'w') as f:
        f.write(version + '\n')

def parse_version(version_str):
    """Parse version string into major, minor, patch"""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$', version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major, minor, patch = map(int, match.groups()[:3])
    prerelease = match.group(4) if match.group(4) else None
    build = match.group(5) if match.group(5) else None
    
    return major, minor, patch, prerelease, build

def bump_version(version_type):
    """Bump version according to semantic versioning"""
    current_version = read_version()
    major, minor, patch, prerelease, build = parse_version(current_version)
    
    if version_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif version_type == 'minor':
        minor += 1
        patch = 0
    elif version_type == 'patch':
        patch += 1
    else:
        raise ValueError(f"Invalid version type: {version_type}")
    
    new_version = f"{major}.{minor}.{patch}"
    if prerelease:
        new_version += f"-{prerelease}"
    if build:
        new_version += f"+{build}"
    
    return new_version

def update_readme_version(version):
    """Update version in README.md"""
    readme_path = Path('README.md')
    if readme_path.exists():
        content = readme_path.read_text()
        # Update version line in README
        content = re.sub(r'\*\*Version:\*\* \d+\.\d+\.\d+', f'**Version:** {version}', content)
        readme_path.write_text(content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python version.py [current|bump-major|bump-minor|bump-patch|set <version>]")
        print("\nCommands:")
        print("  current     - Show current version")
        print("  bump-major  - Bump major version (1.0.0 -> 2.0.0)")
        print("  bump-minor  - Bump minor version (1.0.0 -> 1.1.0)")
        print("  bump-patch  - Bump patch version (1.0.0 -> 1.0.1)")
        print("  set <ver>   - Set specific version")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'current':
        print(f"Current version: {read_version()}")
    
    elif command.startswith('bump-'):
        version_type = command.replace('bump-', '')
        new_version = bump_version(version_type)
        write_version(new_version)
        update_readme_version(new_version)
        print(f"Version bumped to: {new_version}")
    
    elif command == 'set':
        if len(sys.argv) < 3:
            print("Error: Please provide a version number")
            sys.exit(1)
        new_version = sys.argv[2]
        # Validate version format
        try:
            parse_version(new_version)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        write_version(new_version)
        update_readme_version(new_version)
        print(f"Version set to: {new_version}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main() 