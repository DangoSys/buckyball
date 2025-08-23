#!/bin/bash

$1 +permissive +loadmem=$2 +loadmem_addr=800000000 +permissive-off $2 > >(tee $3) 2> >(spike-dasm > $4)