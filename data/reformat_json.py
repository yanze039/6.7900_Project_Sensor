import json
import numpy as np
import matplotlib.pyplot as plt

"""
Here we regroup the data. the data structures are as follows:

"index": {
        "seq": "GGGGCCGGTTCCCC",
        "seq_idx": 0,
        "ph": 6,
        "analyte": {
            "chlor": {
                "shape_term1": 0.09523286749785577,
                "shape_term2": 0.1313436957671284
            },
            "cd": {
                "shape_term1": 0.08548409980512825,
                "shape_term2": 0.0865755467212293
            },
            "enro": {
                "shape_term1": 0.1795389829438892,
                "shape_term2": 0.5916500134995506
            },
            "semi": {
                "shape_term1": 0.1062206028487549,
                "shape_term2": 0.1265020804754554
            }
        }
    },
"""

with open("raw_data.json", 'r') as fp:
    rdata = json.load(fp)

ndata = {}
for kk, item in rdata.items():
    rname = str(kk).split("_")
    name = "_".join(rname[:2])
    if not (name in ndata):
        ndata[name] = {
            "seq": item["seq"],
            "seq_idx": int(rname[0]),
            "ph": item["ph"],
            "analyte": {}
        }
    ndata[name]["analyte"].update(
        {item["analyte"]: {"shape_term1": item["shape_term1"], "shape_term2": item["shape_term2"]}}
    )

with open("data.json", "w") as fp:
    json.dump(ndata, fp, indent=4)



for rr, item in ndata.items():
    for analyte in ["chlor", "cd", "enro", "semi"]:
        ndata[rr]["analyte"][analyte]["shape_term1"] = item["analyte"][analyte]["shape_term1"] * 0.0113

with open("normalized_data.json", "w") as fp:
    json.dump(ndata, fp, indent=4)


response1 = []
response2 = []

for condi, item in ndata.items():
    for analyte in ["chlor", "cd", "enro", "semi"]:
        response1.append(item["analyte"][analyte]["shape_term1"])
        response2.append(item["analyte"][analyte]["shape_term2"])

# print(response1)
# print(response2)

print(f"for wavelength shift, mean: {np.mean(response1)}, std: {np.std(response1)}")
print(f"for intensity change, mean: {np.mean(response2)}, std: {np.std(response2)}")

# visualize the distribution of two types of responses.
# they have different distribution, we just simply normalize the first response.

plt.figure()
plt.hist(response1)
plt.title("distribution of wavelength shift")
plt.savefig("shift.png")

plt.figure()
plt.hist(response2)
plt.title("distribution of intensity change")
plt.savefig("intensity.png")
