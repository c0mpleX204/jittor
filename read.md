计图（Jittor） 深度学习框架

Jittor 是一个基于即时编译和元算子的高性能深度学习框架，整个框架在即时编译的同时，还集成了强大的Op编译器和调优器，为您的模型生成定制化的高性能代码。Jittor还包含了丰富的高性能模型库，涵盖范围包括：图像识别、检测、分割、生成、可微渲染、几何学习、强化学习等。

Jittor前端语言为Python，使用了主流的包含模块化和动态图执行的接口设计，后端则使用高性能语言进行了深度优化。更多关于Jittor的信息可以参考：

    Jittor官网
    Jittor教程
    Jittor模型库
    Jittor文档
    Github开源仓库, Gitee开源仓库

Jittor 安装

Jittor框架对环境要求如下:

    操作系统: Ubuntu >= 16.04 或 Windows Subsystem of Linux（WSL）
    Python：版本 >= 3.7
    C++编译器 （需要下列至少一个）
        g++ （>=5.4.0）
        clang （>=8.0）
    GPU 编译器（可选）：nvcc >=10.0
    GPU 加速库（可选）：cudnn-dev (cudnn开发版, 推荐使用tar安装方法，参考链接)

如果您不希望手动配置环境，我们推荐使用 Docker 进行安装。
除此之外，您还可以使用 pip 安装和手动安装。

注意：目前Jittor通过WSL的方式在Windows操作系统上运行，WSL的安装方法请参考微软官网，WSL版本目前尚不支持CUDA。
Docker 安装

我们提供了Docker安装方式，免去您配置环境的麻烦。Docker安装方法如下：

    # linux CPU only
    docker run -it --network host jittor/jittor
    # linux CPU and CUDA
    docker run -it --network host --gpus all jittor/jittor-cuda
    # mac/windows
    docker run -it -p 8888:8888 jittor/jittor

关于Docker安装的详细教程，可以参考Windows/Mac/Linux通过Docker安装计图
Pip 安装

如果您没有准备好环境，或者使用的不是Ubuntu操作系统， 推荐使用docker安装。如果您已经装好编译器和对应版本的Python,我们强烈推荐您使用Pip安装方法
(如果无法访问github, 可以通过jittor主页下载):

    sudo apt install python3.7-dev libomp-dev
    python3.7 -m pip install jittor
    # or install from github(latest version)
    # python3.7 -m pip install git+https://github.com/Jittor/jittor.git
    python3.7 -m jittor.test.test_example

如果测试运行通过，恭喜你已经安装完成。
jittor会自动在路径中寻找合适的编译器, 如果您希望手动指定编译器, 请使用环境变量 cc_path 和 nvcc_path(可选)。
手动安装

我们将逐步演示如何在Ubuntu 16.04中安装Jittor，其他Linux发行版也可以使用类似的命令进行安装。
目前官方支持的操作系统为 Ubuntu，使用其他操作系统运行 Jittor 可能存在问题。
步骤一：选择您的后端编译器

    # g++
    sudo apt install g++ build-essential libomp-dev
    # OR clang++-8
    wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install_llvm.sh > /tmp/llvm.sh
    bash /tmp/llvm.sh 8

步骤二：安装Python和python-dev

Jittor需要python的版本>=3.7。

    sudo apt install python3.7 python3.7-dev

步骤三：运行Jittor

接下来将通过pip安装jittor

    git clone https://github.com/Jittor/jittor.git
    sudo pip3.7 install ./jittor
    export cc_path="clang++-8"
    # if other compiler is used, change cc_path
    # export cc_path="g++"
    # export cc_path="icc"
    # run a simple test
    python3.7 -m jittor.test.test_example

如果通过了测试，那么您的Jittor已经准备就绪。
可选步骤四：启用CUDA

在Jittor中使用CUDA非常简单，只需设置环境值nvcc_path，如果没有设置该环境变量，Jittor会使用默认环境变量/usr/local/cuda/bin/nvcc去寻找 nvcc。

    # replace this var with your nvcc location 
    export nvcc_path="/usr/local/cuda/bin/nvcc" 
    # run a simple cuda test
    python3.7 -m jittor.test.test_cuda 

如果测试通过，则可以通过设置use_cuda标识符在Jittor中启用CUDA。

    import jittor as jt
    jt.flags.use_cuda = 1

可选步骤五：测试训练Resnet18

您可以通过运行Resnet18训练测试来检查Jittor的完整性。

    python3.7 -m jittor.test.test_resnet

如果出现测试失败或碰到任何问题，欢迎随时联系我们或提交错误报告（issue）， 联系方式如下：