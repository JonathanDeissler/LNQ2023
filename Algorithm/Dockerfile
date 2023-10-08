FROM python:3.9-slim
RUN apt-get update && apt-get install -y git

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output/images/mediastinal-lymph-node-segmentation/ \
    && chown -R user:user /opt/app /input /output

RUN mkdir -p /nnUNet_results/ /nnUNet_raw/ /nnUNet_preprocessed/ \
    && chown -R user:user /nnUNet_results/ /nnUNet_raw/ /nnUNet_preprocessed/
RUN mkdir -p /temporary/inputs/ /temporary/outputs/ /temporary/outputs_pp \
    && chown -R user:user /temporary/inputs/ /temporary/outputs/ /temporary/outputs_pp

COPY /results/ /nnUNet_results/
ENV nnUNet_results /nnUNet_results/
ENV nnUNet_raw /nnUNet_raw/
ENV nnUNet_preprocessed /nnUNet_preprocessed/


USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"


RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt


COPY --chown=user:user process_mvseg.py /opt/app/
COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
