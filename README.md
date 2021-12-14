# Ant Simulation

The project is compiled into a exe for you to run, and also details of how to load the project in Visual Studio 2019 are below.

GitHub: https://github.com/JamesB9/Ant-Simulation

Dependencies:
    Running the exe:
        - NVIDA GPU for CUDA to run.
        - Windows x86 OS
    Loading the project in Visual Studio 2019:
        - The project is built in Visual Studio 2019, and is pre-configured as follows:
            - Load project from "AntSimulation.sln"
            - C++ 20 is required.
            - You must have the CUDA development package installed and configured for visual studio.
                - https://developer.nvidia.com/cuda-downloads
            - Also SFML 2.5.1 is configured to be in your Visual Studio 2019 directory under $(VisualStudioDir)/Libraries/SFML-2.5.1
                - https://www.sfml-dev.org/download/sfml/2.5.1/
                
Verified systems specifications:
    - i9 9900k, 64GB DDR4, RTX 2080Ti 12GB
    - i5 6600k, 8GB DDR4, GTX 1060 6GB
    - ryzen 7 3700x, 16gb ram , GTX 980 8GB
    - ryzen 7 3700x, 16gb ram , GTX 1060 6GB
