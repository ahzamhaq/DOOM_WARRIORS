import unittest

import pandas as pd

from src.config import NORM_COLUMNS
from src.normalize import normalize_records


def run(names, addrs=None, index=None):
    addrs = addrs if addrs is not None else [""] * len(names)
    ids = index or [f"S2-{i}" for i in range(len(names))]
    df = pd.DataFrame({"entity_id": ids, "business_name": names, "business_address": addrs, "country": "X"})
    return normalize_records(df.set_index("entity_id", drop=False))


def col(names, c, addrs=None):
    return run(names, addrs)[c].tolist()


class TestNames(unittest.TestCase):
    def test_ordinary_and_punctuation(self):
        self.assertEqual(col(["Heartland Center", "Berna'S  Massage!", "Goins & Greene"], "name_norm"),
                         ["heartland center", "berna s massage", "goins and greene"])

    def test_legal_suffixes(self):
        got = col(["Shiva Industries Private Limited", "Adams Labs L.L.C.", "TQ Foods Private (Ltd)",
                   "LLC Thomas Fisheries", "Raj & Co.", "Kara Funez, DMD Corp", "Societe Generale SA"], "name_norm")
        self.assertEqual(got, ["shiva industries", "adams labs", "tq foods", "thomas fisheries", "raj",
                               "kara funez dmd", "societe generale"])

    def test_legal_words_in_middle_kept(self):
        self.assertEqual(col(["Concept Limited Edges Private (India)", "Private Care Clinic"], "name_norm"),
                         ["concept limited edges private india", "private care clinic"])

    def test_name_of_only_legal_words_not_emptied(self):
        self.assertTrue(all(col(["LLC", "Pvt Ltd", "Private Limited"], "name_norm")))

    def test_junk_prefixes_and_tags(self):
        got = col(["-- Shiva Traders", ">> Apex", "*** Radha Sons", "#2 Iris", "[[PARTNERS]] Colonial",
                   "M/s Spark Investments", "India Madras Ore (ID: 91803)"], "name_norm")
        self.assertEqual(got, ["shiva traders", "apex", "radha sons", "2 iris", "partners colonial",
                               "spark investments", "india madras ore"])

    def test_domains_and_handles(self):
        out = run(["aimsons.com", ">> DATA.COM", "www.rajco.co.in", "@barretocardiology", "@verra [#20434]",
                   "@ Home 2 All", "Raj & Co. | www.rajco.com", "Bangalore. Pharma L.L.P."])
        self.assertEqual(out["name_norm"].tolist(), ["aimsons", "data", "rajco", "barretocardiology", "verra",
                                                     "home 2 all", "raj", "bangalore pharma"])
        self.assertEqual(out["name_is_domain"].tolist(), [True, True, True, True, True, False, False, False])

    def test_indic_preserved_and_suffix_stripped(self):
        out = run(["श्री स्मार्ट प्रोजेक्ट्स प्राइवेट लिमिटेड", "સુપ્રીમ એન્ટરપ્રાઇઝિસ પ્રા. લિ.",
                   "கோல்டு மீடியா பிரைவேட் லிமிடெட்"])
        self.assertEqual(out["name_norm"].tolist(), ["श्री स्मार्ट प्रोजेक्ट्स", "સુપ્રીમ એન્ટરપ્રાઇઝિસ", "கோல்டு மீடியா"])
        self.assertEqual(out["name_script"].tolist(), ["indic"] * 3)

    def test_unicode_accents_and_scripts(self):
        out = run(["Shri Primax Émbedded Prívate Limited", "Loa Ínc", "東京商事", ""])
        self.assertEqual(out["name_norm"].tolist(), ["shri primax embedded", "loa", "東京商事", ""])
        self.assertEqual(out["name_script"].tolist(), ["latin", "latin", "other", "other"])


class TestAddresses(unittest.TestCase):
    def test_addr_norm(self):
        got = col(["a"] * 4, "addr_norm", ["1175 ELIZA ST, NULL, CITY OF GREEN BAY, WI",
                                           "3224 Rhoades Avenue, South Haven, Minnesota",
                                           "H No 14, New Delhi, Delhi", "12 Rue de la Paix, Paris"])
        self.assertEqual(got, ["1175 eliza st city of green bay wi", "3224 rhoades ave s haven mn",
                               "h no 14 new delhi dl", "12 rue de la paix paris"])

    def test_indic_address_kept(self):
        self.assertEqual(col(["a"], "addr_norm", ["BHIWANDI, महाराष्ट्र"]), ["bhiwandi महाराष्ट्र"])

    def test_empty_and_null(self):
        out = run(["a", "b", "c", "d"], ["", "   ", "<NULL>", "null, null"])
        self.assertEqual(out["addr_empty"].tolist(), [True] * 4)
        self.assertEqual(out["addr_norm"].tolist(), [""] * 4)

    def test_house_numbers(self):
        got = col(["a"] * 11, "house_no", [
            "11513 Fallsburg Road, OH", "OH, 11513 Fallsburg Road", "##177 EMINENCE DR", "100. NEWHALL ST",
            "Plot - 459, Sahi", "DOOR NO 4/1 , AMS HOUSE", "02206 DEBORAH DR", "2Nd Floor, Arjun Nagar",
            "B3/626 V/ A, LUCKNOW", "", "NO G-783 FLOOR, NO 116 MURUGAN ST"])
        self.assertEqual(got, ["11513", "11513", "177", "100", "459", "4/1", "2206", "", "", "", "116"])


class TestFrame(unittest.TestCase):
    def test_columns_index_order_raw_untouched(self):
        ids = ["S3-9", "S2-1", "S3-5"]
        names, addrs = ["Zed LLC", "Alpha Inc", None], ["1 A St", None, "NULL"]
        out = run(names, addrs, index=ids)
        self.assertEqual(out.index.tolist(), ids)
        self.assertEqual(out["entity_id"].tolist(), ids)
        self.assertEqual(out["business_name"].tolist()[:2], names[:2])
        self.assertTrue(set(NORM_COLUMNS) <= set(out.columns))
        self.assertEqual(list(out.columns[-len(NORM_COLUMNS):]), NORM_COLUMNS)
        self.assertFalse(out[NORM_COLUMNS].isna().any().any())
        self.assertEqual(out["name_norm"].tolist(), ["zed", "alpha", ""])

    def test_deterministic_and_slice_independent(self):
        import src.normalize as n
        names = ["Goins & Greene Inc", "@handle", "प्राइवेट लिमिटेड ट्रेडर्स"] * 5
        a = run(names, ["1 Main Street, TX"] * 15)
        old, n._SLICE = n._SLICE, 4
        try:
            b = run(names, ["1 Main Street, TX"] * 15)
        finally:
            n._SLICE = old
        pd.testing.assert_frame_equal(a, b)

    def test_empty_frame(self):
        out = run([], [])
        self.assertEqual(len(out), 0)
        self.assertTrue(set(NORM_COLUMNS) <= set(out.columns))


if __name__ == "__main__":
    unittest.main()
