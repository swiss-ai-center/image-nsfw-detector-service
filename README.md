# Core Engine service for Not Safe For Work image detection

This repository contains the Python + FastAPI code to run a Core Engine service for NSFW detection. It was created from the *template to create a service without a model or from an existing model* available in the repository templates. See <https://docs.swiss-ai-center.ch/how-to-guides/how-to-create-a-new-service> and <https://docs.swiss-ai-center.ch/tutorials/implement-service/>

This service takes as input an image and returns a json with information about the probability that it includes NSFW content.

## NSFW content detection

NSFW stands for *not safe for work*. This Internet slang is a general term associated to un-appropriate content such as nudity, pornography etc. See e.g. https://en.wikipedia.org/wiki/Not_safe_for_work. It is important to exercise caution when viewing or sharing NSFW images, as they may violate workplace policies or community guidelines.

The current service encapsulates a trained AI model to detect NSFW images with a focus on sexual content. Caution: the current version of the service is not able to detect profanity and violence for now.

### Definition of categories

The border between categories is sometimes thin, e.g. what can be 
considered as acceptable nudity in some cultural context would be considered as 
pornography by others. Therefore we need to disclaim any complaints that would
be done by using the model trained in this project. We can't be taken responsible
of any offense or classifications that would be falsely considered as appropriate 
or not. To make the task even more interesting, we went here for two main 
categories *nsfw* and *safe* in which we have sub-categories.

- **nsfw**:
  - **porn**: male erection, open legs, touching breast or genital parts, 
  intercourse, blowjob, etc; men or women nude and with open legs fall into
  this category; nudity with sperma on body parts is considered porn
  - **nudity**: penis visible, female breast visible, vagina visible in 
  normal position (i.e. standing or sitting but not open leg)
  - **suggestive**: images including people or objects making someone think 
  of sex and sexual relationships; genital parts are not visible otherwise
  the image should be in the porn or nudity category; dressed people kissing 
  and or touching fall into this category; people undressing; licking 
  fingers; woman with tong with sexy bra
  - **cartoon_sex**: cartoon images that are showing or strongly 
  suggesting sexual situation
- **safe**:
  - **neutral**: all kind of images with or without people not falling 
  into porn, nudity or suggestive category
  - **cartoon_neutral**: cartoon images that are not showing or  
  suggesting sexual situation

Inspecting the output giving probabilities for the categories (safe vs not-safe) and
the sub-categories, the user can decide where to place the threshold on what is 
acceptable or not for a given service.


### Data set used to build the model

A dataset was assembled using existing NSFW image sets and was completed with web scraping data.
The dataset is available for research purpose - contact us if you want to have an access. Here
are some statistics about its conent (numbers indicate amount of images). The dataset is balanced among
the categories, which should avoid biased classifications.

| categories     | safe    |        |         | nsfw       |        |      |         | total   |       |       |
|----------------|---------|--------|---------|------------|--------|------|---------|---------|-------|-------|
| sub-categories | general | person | cartoon | suggestive | nudity | porn | cartoon | safe    | nsfw  | all   |
| v2.2           | 5500    | 5500   | 5500    | 5500       | 5500   | 5500 | 5500    | 16500   | 22000 | 38500 |

### Model training and performance

We used transfer learning on MobileNetV2 which present a good trade-off between performance and runtime efficiency.

| Set  | Model                                                   | Whole |       | Val   |       | Test  |       |
|------|---------------------------------------------------------|-------|-------|-------|-------|-------|-------|
|      |                                                         | sa/ns | sub   | sa/ns | sub   | sa/ns | sub   |
| V2.1 | TL_MNV2_finetune_224_B32_AD1E10-5_NSFW-V2.1_DA2.hdf5    |       |       | 95.7% | 85.1% | 95.7% | 86.1% |

In this Table, the performance is reported as accuracy on the safe vs not-safe (sa/ns) main categories and
on the sub-categories (sub). The sub performance in indeed lower as we have naturally more confusion between
some categories and as there is simply a larger cardinality in the number of classes.


## How to test locally the service?

1. Create and activate the virtual environment:
```sh
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Then install the dependencies:
```sh
pip install --requirement requirements.txt
pip install --requirement requirements-all.txt
```

3. Run locally an instance of the Core AI Engine. For this follow the installation 
instructions available here: https://docs.swiss-ai-center.ch/reference/core-engine/. Here are
the steps:
  - Get the core engine code from here: https://github.com/swiss-ai-center/core-engine/tree/main
  - Backend: follow instructions in section `Start the service locally with Python`, in a first
    terminal start the dependencies with `docker compose up` and in a second terminal in the `src`
    sub-directory start the application with `uvicorn --reload --port 8080 main:app`. The backend 
    api should be visible in the browser.
  - This service: in a terminal start the service with `cd src` and 
    `uvicorn main:app --reload --host localhost --port 9090`. The service should register to the
    Core Engine backend and now be visible on the api page.
  - Frontend: in a terminal follow the starting instruction (make sure Nodes and npm are 
    installed).
