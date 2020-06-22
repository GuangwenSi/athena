# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

datadir=$1
cd $datadir

wget http://www.openslr.org/resources/12/dev-clean.tar.gz
wget http://www.openslr.org/resources/12/dev-other.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
wget http://www.openslr.org/resources/12/test-other.tar.gz
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
wget http://www.openslr.org/resources/12/train-other-500.tar.gz

echo "extracting dev-clean..."
tar xzf dev-clean.tar.gz
echo "extracting dev-other..."
tar xzf dev-other.tar.gz
echo "extracting test-clean..."
tar xzf test-clean.tar.gz
echo "extracting test-other..."
tar xzf test-other.tar.gz
echo "extracting train-clean-100..."
tar xzf train-clean-100.tar.gz
echo "extracting train-clean-360..."
tar xzf train-clean-360.tar.gz
echo "extracting train-other-500..."
tar xzf train-other-500.tar.gz

cd -
