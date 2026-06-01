#!/usr/bin/env bash

# Create a reverse SSH tunnel 
NODE1_IP="$1" # e.g. .XXX
PORT="$2"
USERNAME="$3"

ssh -N -R "${PORT}:localhost:${PORT}" "${USERNAME}@${NODE1_IP}"
