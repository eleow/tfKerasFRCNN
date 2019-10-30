[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/eleow/tfKerasFRCNN">
    <img src="misc/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">tfKerasFRCNN</h3>

  <p align="center">
    Faster R-CNN for tensorflow keras, packaged as a library
    <br />
    <a href="https://github.com/eleow/tfKerasFRCNN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/eleow/tfKerasFRCNN">View Demo</a>
    ·
    <a href="https://github.com/eleow/tfKerasFRCNN/issues">Report Bug</a>
    ·
    <a href="https://github.com/eleow/tfKerasFRCNN/issues">Request Feature</a>
  </p>
</p>



## About

Faster RCNN has been implemented to be used as a library, following Tensorflow Keras Model API as much as possible. This consistent interface will allow any user who is already familiar with Tensorflow Keras to use our APIs easily. To simplify the API, only basic configuration options would be available.

In order to make things easier for the user, we have also included useful features such as automatic saving of model and csv during training, as well as automatic continuation of training. This is especially useful if running in Google Colab GPU, as there are time limits for each session.

## Getting Started

### Pre-requisites and Dependencies

The dependencies for the library is shown in the figure below.
![Dependencies](misc/pydeps.png)

Created via [pydeps](https://pydeps.readthedocs.io/en/latest/) using command
```
pydeps FRCNN.py --max-bacon=4 --cluster)
```

### Installation
See "conda setup.txt" for installation instructions

## Usage

See [_TrainAndTestFRCNN.py](https://github.com/eleow/tfKerasFRCNN/blob/master/_TrainAndTestFRCNN.py) or [_TrainAndTestFRCNN.ipynb](https://github.com/eleow/tfKerasFRCNN/blob/master/_TrainAndTestFRCNN.ipynb) for end-to-end example of how to use the library


## License
Distributed under the [MIT License](LICENSE)

## Acknowledgements
Code was modified and refactored from original code by [RockyXu66](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras) and [kbardool](https://github.com/kbardool/keras-frcnn)


<div>Shoes icon by <a href="https://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/"         title="Flaticon">www.flaticon.com</a></div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/eleow/tfKerasFRCNN
[contributors-url]: https://github.com/eleow/tfKerasFRCNN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/eleow/tfKerasFRCNN
[forks-url]: https://github.com/eleow/tfKerasFRCNN/network/members
[stars-shield]: https://img.shields.io/github/stars/eleow/tfKerasFRCNN
[stars-url]: https://github.com/eleow/tfKerasFRCNN/stargazers
[issues-shield]: https://img.shields.io/github/issues/eleow/tfKerasFRCNN
[issues-url]: https://github.com/eleow/tfKerasFRCNN/issues
[license-shield]: https://img.shields.io/github/license/eleow/tfKerasFRCNN
[license-url]: https://github.com/eleow/tfKerasFRCNN/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/edmundleow
[product-screenshot]: images/screenshot.png

