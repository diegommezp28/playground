FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Avoid warnings by switching to noninteractive
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y curl libpq-dev gcc