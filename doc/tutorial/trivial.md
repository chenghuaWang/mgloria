Trivial
===

## Clangd
The clangd cuda support is not quite good. If you want to edit *.cu or *.cuh file or you want clangd to recognize __CUDACC__ macro. You should add some args in the clangd setting files.

Basically, you can add args to `.vscode/setting.json` below if you are using clangd-12 and your cuda is installed correctly into `/usr/local/cuda`. You may use the NVIDIA NSight to Debug. Unfortunately, the NSight using Microsoft c++ intelliSenseEngine as backend which can cause crash with clangd, you should disable it.

```json
    "clangd.path": "/bin/clangd-12",
    "clangd.arguments": [
        "-xcuda",
        "--cuda-path=/usr/local/cuda"
    ],
    "C_Cpp.intelliSenseEngine": "Disabled"
```

## C++20
The c++ volatile which I heavily depends on has changed. According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1152r1.html. I don't want to adapt this new law anymore..:-( . The c++ standard I used is back to c++14, it also works well.
