from os import makedirs
from os.path import exists
import wget


if __name__ == "__main__":
    data_direct_links = {
        "human_annotations": {
            "sst_preprocessed.pickle":      "https://drive.google.com/uc?export=download&id=1UJcd8KMIIQTwd8xHKX_53aBRhNQfNoFR",
            "esnli_preprocessed_2.pickle":  "https://drive.google.com/uc?export=download&id=1LdpnR6IoKH__jIJzYPv-2auRbxGrYtk3",
            "esnli_preprocessed_3.pickle":  "https://drive.google.com/uc?export=download&id=1pGjErLGln4v0RVnSJKOfMtcySJB0_UtG",
            "multirc_preprocessed.pickle":  "https://drive.google.com/uc?export=download&id=1yEdjiGJlu41s6_rxZPnl-9IkAKETux6w"
        },
        "masked_examples": {
            "masked_examples_dev_sst-2.pickle":          "https://drive.google.com/uc?export=download&id=1ETfhXdPGN0972NbTFwNHvbC1FbcOt6NI",
            "masked_examples_dev_sst.pickle":            "https://drive.google.com/uc?export=download&id=1JwwAs6IFZZYil5oR-O2P12-1_uJ0tSkD",
            "masked_examples_dev_esnli.pickle":          "https://drive.google.com/uc?export=download&id=115C_b9sri3vskF1lqqdZkhGXtakdDwZP",
            "masked_examples_dev_multirc_split0.pickle": "https://drive.google.com/uc?export=download&id=17wZQUfuQZz7rjbMBqTx9AqxFkTSRPXec",
            "masked_examples_dev_multirc_split1.pickle": "https://drive.google.com/uc?export=download&id=1SsG4WsIVEx0yWt1j5zehIXMapzpkL09L",
            "masked_examples_dev_multirc_split2.pickle": "https://drive.google.com/uc?export=download&id=1zE4R_O-4qaWwBaXmp2kmiG6bHt_YtWeL",
            "masked_examples_train_sst-2.pickle":        "https://drive.google.com/uc?export=download&id=1QiUrIomhmH36d90NGCYcKb_D2HkZqeee",
            "masked_examples_train_sst.pickle":          "https://drive.google.com/uc?export=download&id=1JQ-WVW6tdVv44MTgjaqkFptfYr40e5JB",
        }
    }

    destination = '../data/pickle_files/'

    for path, files in data_direct_links.items():
        for file_name, direct_url in files.items():
            out_dir = destination + path
            if not exists(out_dir):
                makedirs(out_dir)

            print("Downloading file `{}`...".format(file_name))
            wget.download(direct_url, out=out_dir)



