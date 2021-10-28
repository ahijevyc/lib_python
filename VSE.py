import glob
import os
import sys

workdir = "/glade/work/ahijevyc/share/VSE/"

def composites():
    # get all files ending with .00z.txt
    ff = glob.glob(workdir + "*z.txt")
    # split filename by periods and get first part
    cl = [os.path.basename(f).split('.')[0] for f in ff]
    return set(cl)

def pointcollections():
    return ["shr10_700 max", "torn max"]

def pointcollection(collection, coord="north"):
    pdict = {}
    # Ran NARR_composite.py with all LTC categories except near coast and well inland,
    # 0003z, 0609z, 1215z, 1821z combined 
    # for north pointing up coordinate system these are points in the composite corresponding to max shear or torn rpts
    if coord == "north":
        pdict["shr10_700 max"] = ["shear max/58deg/280km","TC center/0deg/0km","opposite shr max/238deg/280km"]
        pdict["torn max"] = ["torn max/74.3deg/254.4km","TC center/0deg/0km","opposite torn max/254.3deg/254.4km"]
        pdict["tornadoes_well_inland torn max"] = ["torn max/80deg/200km","TC center/0deg/0km","opposite torn max/260deg/200km"]
    else:
        print(f"VSE: Unexpected coord {coord}")
        sys.exit(1)

    if collection in pdict:
        points = pdict[collection]
    else:
        print(f"VSE: Unexpected collection {collection} for {coord} coord")
        sys.exit(1)

    return points



