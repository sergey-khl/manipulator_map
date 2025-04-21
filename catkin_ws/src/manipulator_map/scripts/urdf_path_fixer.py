#!/usr/bin/env python3
"""
URDF path fixer

This script takes an input URDF file and replaces all `package://.../meshes/...<filename>` references
with local `./meshes/<filename>` references. Standardizes robot definitions for this project.

Of course, this means that robots will be defined with 1 urdf file and a meshes folder in that directory.
Take a look at how other robots to import your own robot.

Usage:
    python urdf_path_fixer.py input.urdf [output.urdf]

If `output.urdf` is not provided, the input file will be overwritten.
"""
import argparse
import sys
import xml.etree.ElementTree as ET

# adjusts package:// references to work with this repo
def fixPaths(tree: ET.ElementTree) -> str:
    root = tree.getroot()
    robot_name = root.attrib.get('name')
    if not robot_name:
        print("Error: <robot> tag with a 'name' attribute not found.")
        return False

    updated = False
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename', '')
        if 'meshes/' in filename:
            # Extract just the filename after the last '/'
            fname = filename.split('/')[-1]
            new_path = f"package://manipulator_map/robots/{robot_name}/meshes/{fname}"
            mesh.set('filename', new_path)
            updated = True
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Localize mesh references in URDF to ./meshes folder.")
    parser.add_argument(
        'input_file',
        help='Path to the input URDF file')
    parser.add_argument(
        'output_file',
        nargs='?',
        default=None,
        help='Path to the output URDF file (defaults to input file)')
    args = parser.parse_args()

    in_path = args.input_file
    out_path = args.output_file or in_path

    try:
        tree = ET.parse(in_path)
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Failed to load URDF '{in_path}': {e}")
        sys.exit(1)

    if not fixPaths(tree):
        print("No mesh references updated.")
        sys.exit(0)

    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"Updated URDF written to: {out_path}")

if __name__ == '__main__':
    main()
