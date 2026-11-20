import tempfile
import unittest
from pathlib import Path

from resonoise.database import Database
from resonoise.errors import NotFoundError


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, ref=False):
        return self.db.add_recording(name="clip", source="upload", original_filename="clip.wav", sample_rate=16000,
                                     original_sample_rate=44100, channels=2, duration=3.0, size_bytes=100,
                                     audio_bytes=b"RIFFxxxx", container="wav", reference_bytes=b"ref" if ref else None)

    def test_recording_roundtrip_and_list(self):
        rec = self.add(ref=True)
        got = self.db.get_recording(rec["id"])
        self.assertEqual(got["name"], "clip")
        self.assertTrue(Path(got["audio_path"]).exists() and Path(got["reference_path"]).exists())
        self.assertEqual([r["id"] for r in self.db.list_recordings()], [rec["id"]])

    def test_analysis_processed_and_labels(self):
        rid = self.add()["id"]
        self.assertIsNone(self.db.get_analysis(rid))
        self.db.save_analysis(rid, {"a": 1}, {"p": 2}, "fan", 0.9)
        self.db.save_analysis(rid, {"a": 3}, {"p": 4}, "ac", 0.8)          # overwrite
        a = self.db.get_analysis(rid)
        self.assertEqual((a["features"], a["payload"]), ({"a": 3}, {"p": 4}))
        pid = self.db.add_processed(rid, profile="ac", params={"x": 1}, metrics={"m": 1}, payload={"k": 1}, audio_bytes=b"abc")
        self.assertEqual(self.db.get_processed(pid)["payload"]["processed_id"], pid)
        self.assertEqual(self.db.list_recordings()[0]["n_processed"], 1)
        self.assertEqual(self.db.list_recordings()[0]["source_class"], "ac")
        self.db.set_label(rid, "fan", {"f": 1})
        self.db.set_label(rid, "ac", {"f": 2})                              # replaces
        self.assertEqual(self.db.get_label(rid), "ac")
        self.assertEqual(len(self.db.list_labels()), 1)
        self.assertEqual(self.db.stats(), {"recordings": 1, "analyses": 1, "processed": 1, "labels": 1})

    def test_delete_cascades_and_removes_files(self):
        rid = self.add(ref=True)["id"]
        pid = self.db.add_processed(rid, profile="x", params={}, metrics={}, payload={}, audio_bytes=b"abc")
        paths = [Path(self.db.get_recording(rid)["audio_path"]), Path(self.db.get_processed(pid)["audio_path"])]
        self.db.save_analysis(rid, {}, {}, None, None)
        self.db.delete_recording(rid)
        self.assertFalse(any(p.exists() for p in paths))
        self.assertEqual(self.db.stats(), {"recordings": 0, "analyses": 0, "processed": 0, "labels": 0})
        with self.assertRaises(NotFoundError):
            self.db.get_recording(rid)

    def test_delete_processed_only(self):
        rid = self.add()["id"]
        pid = self.db.add_processed(rid, profile="x", params={}, metrics={}, payload={}, audio_bytes=b"abc")
        self.db.delete_processed(pid)
        self.assertEqual(self.db.list_processed(rid), [])
        self.db.get_recording(rid)
        with self.assertRaises(NotFoundError):
            self.db.get_processed(pid)

    def test_persists_across_instances(self):
        rid = self.add()["id"]
        self.assertEqual(Database(Path(self.tmp.name)).get_recording(rid)["id"], rid)


if __name__ == "__main__":
    unittest.main()
