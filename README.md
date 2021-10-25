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
    Pham, Bui, Mai, Nguyen (2021). Double Trouble: How to not explain a text classifier's decisions using counterfactuals synthesized by masked language models.
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


<!-- USAGE EXAMPLES -->

## Usage

- Generate masked examples for SST-2, SST, e-SNLI and MultiRC datasets.
- Pre-process human highlights for SST, e-SNLI and MultiRC.
- Run attribution methods reported in our paper: 
  
    Notes: 
    - Replace `TASK_NAME` with one of the following tasks: `SST-2`, `SST`, `ESNLI`, `MultiRC`.
    - Replace `METRIC` with one of the following tasks: `auc`, `auc_bert`, `roar`, `roar_bert`, `human_highlights`.
    
    - Input Marginalization (IM)
      ```sh
      bash script/run_analyzers.sh TASK_NAME METRIC InputMargin
      ```
    - Leave One Out variants: LOO<sub>empty</sub>, LOO<sub>unk</sub>, LOO<sub>zero</sub>
      ```sh
      bash script/run_analyzers.sh TASK_NAME METRIC OccEmpty
      bash script/run_analyzers.sh TASK_NAME METRIC OccUnk
      bash script/run_analyzers.sh TASK_NAME METRIC OccZero
      ```
    - LIME and LIME<sub>BERT</sub>
      ```sh
      bash script/run_analyzers.sh TASK_NAME METRIC LIME
      bash script/run_analyzers.sh TASK_NAME METRIC LIME-BERT
      ``` 
- Evaluation
    - Deletion and BERT-based Deletion (AUC vs. AUC<sub>rep</sub>)
    - RemOve And Retrain (ROAR)
    - Agreement with human-annotated highlights
    - Sanity check
- Visualization for attribution maps (binary & real-valued)
  ```sh
  python run_demo.py --text_a "Two men dressed in black practicing martial arts on a gym floor ." 
                     --text_b "Two men are doing martial arts ."
                     --classifier "ESNLI"
  ```

[![ESNLI example][project-example-esnli]](https://github.com/anguyen8/im/)

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