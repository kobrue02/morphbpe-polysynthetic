from pathlib import Path

import hfst

OUT_PATH = Path(__file__).resolve().parent / "grn.seg.hfst"


def load_or_build():
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} is missing. See tools/apertium-grn/BUILD.md to rebuild it "
            "(requires the real Apertium/lttoolbox/HFST-CLI/VISLCG3 toolchain)."
        )
    istr = hfst.HfstInputStream(str(OUT_PATH))
    fst = istr.read()
    istr.close()
    return fst
