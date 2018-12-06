1. 运行环境
	操作系统：window10
	处理器：i5 4核
	内存：12GB

2. 主要依赖包
	编程语言：anaconda下 Python 3.6.2
	主要依赖包：tensorflow 1.7.0
				numpy 1.14.2
				os
				pickle
				time
				
3. 目录结构
	ruijin_project/
		code/*
		data/
			ruijin_round1_test_a_20181022/*
			ruijin_round1_test_b_20181112/*
			ruijin_round1_train_20181022/
				ruijin_round1_train2_20181022/*
			...
		submit/
			time1/*
			time2/*
			...
		readme.txt
	
4. 运行
	进入code文件夹下，使用命令 python main.py
	或在根文件下，使用命令 python code/main.py
	
重要说明：
评测时直接把官网上面压缩文件解压到data文件下即可，因为从官网下载的训练集解压后还有一层文件夹，由于官方说要求文件形式和官网一致，所以train文件夹下还有一层文件夹，参考上面的ruijin_round1_train2_20181022/*，
另外，提交的代码只用于评测B榜数据。由于随机数没有设置种子，多次运行会产生多个结果，我把多次运行的结果按照时间建立文件夹，分别为submit文件夹下的time1，time2 ... ，
其中每个文件夹下就是一次运行结果，例如 time1 文件夹下包含所有的预测ann文件，time2 文件夹也是如此，评测取均值即可。
为了防止意外，在根文件夹下，我预先放入了之前提交的结果pre_submit.zip，可以参考，如有问题，请联系chenwi4323@gmail.com。
