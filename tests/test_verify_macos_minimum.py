import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_macos_minimum import (
    InspectionError,
    app_minimum,
    find_violations,
    is_macho,
    minimum_os_from_otool,
    parse_version,
)

# Trimmed `otool -l` output of the macosx_15_0 libmlx.dylib that broke #531.
OTOOL_BUILD_VERSION = """
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 15.0
      sdk 15.5
   ntools 1
Load command 10
      cmd LC_SOURCE_VERSION
  cmdsize 16
  version 0.0
"""

OTOOL_VERSION_MIN = """
Load command 8
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 12.0
      sdk 13.1
"""


class VersionTests(unittest.TestCase):
    def test_trailing_zeros_compare_equal(self):
        self.assertEqual(parse_version('14.4.0'), parse_version('14.4'))
        self.assertGreater(parse_version('15.0'), parse_version('14.4.0'))
        self.assertLess(parse_version('14.0'), parse_version('14.4'))

    def test_reads_the_app_minimum_from_package_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'package.json'
            package.write_text(json.dumps({'build': {'mac': {'minimumSystemVersion': '14.4.0'}}}))
            self.assertEqual(app_minimum(str(package)), (14, 4))


class OtoolParsingTests(unittest.TestCase):
    def test_reads_build_version_minos(self):
        self.assertEqual(minimum_os_from_otool(OTOOL_BUILD_VERSION), (15,))

    def test_reads_the_older_version_min_command(self):
        self.assertEqual(minimum_os_from_otool(OTOOL_VERSION_MIN), (12,))

    def test_source_version_is_not_mistaken_for_a_minimum(self):
        self.assertEqual(minimum_os_from_otool(OTOOL_BUILD_VERSION.replace('minos 15.0', 'minos 14.0')), (14,))

    def test_fat_binary_uses_the_highest_slice(self):
        self.assertEqual(minimum_os_from_otool(OTOOL_VERSION_MIN + OTOOL_BUILD_VERSION), (15,))

    def test_missing_load_command_returns_none(self):
        self.assertIsNone(minimum_os_from_otool('Load command 0\n      cmd LC_SEGMENT_64\n'))


class FindViolationsTests(unittest.TestCase):
    def test_flags_only_mach_o_files_above_the_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'mlx' / 'lib').mkdir(parents=True)
            new = root / 'mlx' / 'lib' / 'libmlx.dylib'
            old = root / 'libggml.dylib'
            text = root / 'mlx.metallib'
            new.write_bytes(b'\xcf\xfa\xed\xfe' + b'\0' * 28)
            old.write_bytes(b'\xcf\xfa\xed\xfe' + b'\0' * 28)
            text.write_bytes(b'MTLB' + b'\0' * 28)
            minimums = {str(new): (15, 0), str(old): (14,)}

            checked, violations = find_violations(
                [str(root)], (14, 4), read_minimum=lambda path: minimums[path]
            )

        self.assertEqual(checked, 2)
        self.assertEqual(violations, [(str(new), (15, 0))])

    def test_an_unreadable_mach_o_is_reported_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / 'broken.dylib'
            broken.write_bytes(b'\xcf\xfa\xed\xfe' + b'\0' * 28)

            def fail(_path):
                raise InspectionError('truncated or malformed object')

            errors = []
            checked, violations = find_violations([tmp], (14, 4), read_minimum=fail, errors=errors)

        self.assertEqual((checked, violations), (1, []))
        self.assertEqual(errors, [(str(broken), 'truncated or malformed object')])

    def test_recognises_fat64_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, magic in (('fat64', b'\xca\xfe\xba\xbf'), ('fat64_swapped', b'\xbf\xba\xfe\xca')):
                path = Path(tmp) / name
                path.write_bytes(magic + b'\0' * 28)
                self.assertTrue(is_macho(str(path)), name)


if __name__ == '__main__':
    unittest.main()
