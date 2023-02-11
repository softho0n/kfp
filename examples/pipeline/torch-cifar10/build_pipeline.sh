#!/bin/bash
base_dir=$(pwd)
components_dir=$base_dir/components

for component in $components_dir/*/; do
    cd $component && ./build.sh
done

echo "Components Dockerize Done!!"
