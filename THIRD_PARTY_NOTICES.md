# Third-party notices

flashcluster is MIT-licensed (see LICENSE). It includes code derived from the following project.

## FAISS

The e-means implementation in `src/kmeans.rs` is ported from the FAISS reference implementation
(`faiss/contrib/e_means`, https://github.com/facebookresearch/faiss), used under the MIT License:

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NVIDIA CUTLASS (not included)

The optional int8 kernels in `cuda/cutlass_int8*.cu` compile against NVIDIA CUTLASS headers, which are not
included in this repository and are fetched separately. CUTLASS is licensed under the BSD-3-Clause License
(https://github.com/NVIDIA/cutlass).
