<div id="top"></div>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />

<div align="center">
  <!--
  <a href="https://github.com/anguyen8/im">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->


  <h3 align="center">Analysis of Attribution Methods</h3>

  <p align="center">
    <a href="https://arxiv.org/abs/2110.11929">Double Trouble: How to not explain a text classifier's decisions using counterfactuals synthesized by masked language models?</a>
    Pham, Bui, Mai, Nguyen (2021).
    <br />
    <br />
    <a href="https://github.com/anguyen8/im/issues">Report Bug</a>
    ·
    <a href="https://github.com/anguyen8/im/issues">Request Feature</a>
  </p>

</div>



<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>




<!-- ABOUT THE PROJECT -->

## About The Project

The project provides a rigorous evaluation using 5 metrics and 3 human-annotated datasets to better assess the attribution method Input Marginalization and compare with Leave-one-out - a simple yet strong baseline which remove a feature (i.e., token) by simply replacing it with an empty string.
The source code was released for the following publication:

This repository contains source code necessary to reproduce some of the main results in our paper.

**If you use this software, please consider citing:**

    @inproceedings{pham2021double,
        title={Double Trouble: How to not explain a text classifier's decisions using counterfactuals synthesized by masked language models},
        author={Thang Pham, Trung Bui, Long Mai, Anh Nguyen},
        booktitle={arXiv pre-print},
        year={2021}
    }

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

* Anaconda 4.10 or higher
* Python 3.7 or higher
* pip version 21 or higher

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/anguyen8/im.git
   ```

2. Create and activate a Conda environment

   ```sh
   conda create -n im python=3.9
   conda activate im
   ```

3. Install required libraries

   ```sh
   pip install -r requirements.txt
   ```

4. Change the working directory to `src` and export `PYTHONPATH` before running any script.
   
    ```shell
    cd src/
    export PYTHONPATH=/path/to/your/im:/path/to/your/im/src:/path/to/your/im/src/transformers
    
    # For example
    export PYTHONPATH=/home/thang/Projects/im:/home/thang/Projects/im/src:/home/thang/Projects/im/src/transformers
   
    # Optional: Single or multiple GPUs
    # Single GPU
    export CUDA_VISIBLE_DEVICES=0 # or whatever GPU you prefer
   
    # Multiple GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3... # list of GPUs separated by a comma
    ```

5. Download the pre-computed pickle files of masked examples used for attribution methods and human highlights used for evaluation by running the following script

    ```sh
    python auto_download.py
    ```

<!-- USAGE EXAMPLES -->

## Usage

### 1. Reproduce the quantitative results

**Summary** of the features provided (out-of-the-box):

* BERT-based classifiers pre-trained on SST-2, e-SNLI, and MultiRC were uploaded to [HuggingFace](https://huggingface.co/pmthangk09) and will be loaded directly in the code for the corresponding task.
* Generate intermediate masked text (i.e. with MASK) for `LOO` and `IM` (which needs another step to replace the MASK with BERT suggestions in order to generate counterfactuals). 
  * We also provide a _pre-computed_ pickle file of these intermediate masked examples [here](https://drive.google.com/drive/folders/17YpPgUerL_I-smN6Wy2ok4Kuu7fn6ZTx).
* Run attribution methods 
  * (`LOOEmpty`, `LOOUnk` `LOOZero`, `IM`) on `SST-2`, `SST`, `ESNLI`, `MultiRC` datasets.
  * (`LIME`, `LIME-BERT`) on `SST` dataset.
* Evaluate the generated attribution maps (by one of the above methods) using one of the following quantitative metrics: `auc`, `auc_bert`, `roar`, `roar_bert`, `human_highlights`.
  * We provide the pre-processed and quality-controlled human highlights used in the paper (download [here](https://drive.google.com/drive/folders/17iKO0WRCVo_8nd3xz3hcvL310huxml78?usp=sharing)).



Run the following turn-key script to generate quantitative results

```sh
bash ../script/run_analyzers.sh TASK_NAME METRIC ATTRIBUTION_METHOD
```


- Replace `TASK_NAME` with one of the following tasks: `SST-2`, `SST`, `ESNLI`, `MultiRC`.

- Replace `METRIC` with one of the following metrics: `auc`, `auc_bert`, `roar`, `roar_bert`, `human_highlights`.

- Replace `ATTRIBUTION_METHOD` with on of the following methods: `LOOEmpty`, `LOOUnk`, `LOOZero`, `IM`, `LIME`, `LIME-BERT`.

If the selected metric is `roar` or `roar_bert`, after generating attribution maps for `LOO` and `IM`, we need to run the following script to re-train and evaluate new models.

```sh
# Change the directory to /src/transformers before running the script run_glue.sh
cd transformers/ # Assume you are now under src/
bash run_glue.sh
```

<!--
- Evaluation

  - Deletion and BERT-based Deletion (AUC vs. AUC<sub>rep</sub>)
  - RemOve And Retrain (ROAR)
  - Agreement with human-annotated highlights
  - Sanity check
-->

### 2. Visualization for attribution maps (binary & real-valued)

We also provide an interactive demo to compare the qualitative results between `LOOEmpty` and `IM`.

  ```sh
  # Make sure your working directory is src/ before running this script
  # The positional arguments are: task_name text_a text_b theta which is the threshold used to binarize attribution maps (default value is 0.05)
  # For SST
  bash ../scripts/run_demo.sh "SST" "Mr. Tsai is a very original artist in his medium , and What Time Is It There ?" "" 0.05
  
  # For ESNLI
  bash ../scripts/run_demo.sh "ESNLI" "Two men dressed in black practicing martial arts on a gym floor ." "Two men are doing martial arts ." 0.05
  ```
For example, when running the above script for ESNLI, we will get this output:

[![ESNLI example][project-example-esnli-output]](https://github.com/anguyen8/im/)

which is similar to one of our figures (i.e. Fig. 3) shown in the paper.

[![ESNLI example][project-example-esnli]](https://github.com/anguyen8/im/)

For the comparison between `LOOEmpty` and `IM` in terms of real-valued attribution maps, the above script will generate a tex file under the directory [`data/attribution_maps/`](https://github.com/anguyen8/im/tree/main/data/attribution_maps).
We just need to simply convert this file to PDF format for viewing.

[![ESNLI example][project-example-esnli-real-valued-am]](https://github.com/anguyen8/im/)

<!--

- [] Analysis of attribution maps
  - [] Out-of-distribution issue (Sec. 5.1)
  - [] BERT often replaces a word by itself (Sec. 5.2)
  - [] Attribution magnitude (Sec. 5.2)
    -->

See the [open issues](https://github.com/anguyen8/im/issues) for a full list of proposed features (and
known issues).


<!-- CONTRIBUTING -->

<!--

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
   -->

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->

## Contact

Thang Pham - [@pmthangxai](https://twitter.com/pmthangxai) - tmp0038@auburn.edu

Project Link: [https://github.com/anguyen8/im](https://github.com/anguyen8/im)


<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* [Huggingface's Transformers](https://huggingface.co/transformers/)

* [LIME for NLP](https://github.com/marcotcr/lime)

* [Best README template by Othneil Drew](https://github.com/othneildrew/Best-README-Template#about-the-project)

<p align="right">&#40;<a href="#top">back to top</a>&#41;</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/anguyen8/im.svg?style=for-the-badge
[contributors-url]: https://github.com/anguyen8/im/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/anguyen8/im.svg?style=for-the-badge
[forks-url]: https://github.com/anguyen8/im/network/members
[stars-shield]: https://img.shields.io/github/stars/anguyen8/im.svg?style=for-the-badge
[stars-url]: https://github.com/anguyen8/im/stargazers
[issues-shield]: https://img.shields.io/github/issues/anguyen8/im.svg?style=for-the-badge
[issues-url]: https://github.com/anguyen8/im/issues
[license-shield]: https://img.shields.io/github/license/anguyen8/im.svg?style=for-the-badge
[license-url]: https://github.com/anguyen8/im/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/thangpm
[product-screenshot]: images/screenshot.png
[project-example-esnli]: images/example_esnli.png
[project-example-esnli-output]: images/example_esnli_output.png
[project-example-esnli-real-valued-am]: images/example_esnli_output_real_valued_am.png
