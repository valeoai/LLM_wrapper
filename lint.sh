#!/bin/bash

printf "\033[0;32m Launching isort \033[0m\n"
isort llm_wrapper ./*.py

printf "\033[0;32m Launching black \033[0m\n"
black llm_wrapper ./*.py

printf "\033[0;32m Launching flake8 \033[0m\n"
flake8 llm_wrapper ./*.py
