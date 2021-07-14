# 1. numpy

在深度学习中，我们使用不是简单的数字，而往往是矩阵或者是向量，因此numpy会更加的有用。

- np.exp()


$$
np.exp(x) = (e^{x_1}, e^{x_2}, ..., e^{x_n})
$$

  ```python
  import numpy as np
  
  # example of np.exp
  x = np.array([1, 2, 3])
  print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
  ```

- np.reshape()

    更改张量的形状，内部元素不变化

    > **numpy.**reshape**(a,** *newshape***,** *order='C'***)**

    ```python
    a=np.array([[ 0,  1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10, 11]])
    b=np.reshape(a,(3,4))
    print(a)
    print(b)
    #[[ 0  1  2  3  4  5]
    # [ 6  7  8  9 10 11]]
    #[[ 0  1  2  3]
    # [ 4  5  6  7]
    # [ 8  9 10 11]]
    ```

- np.shape()

    返回张量的形状

- np.resize()

    修改张量的size

    1. 特别注意两种resize

       ```python
       #会自动在扩大的位置填充0，在a的基础上进行修改
       a=np.array([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11]])
       a.resize((3,6))
       print(a)
       #[[ 0  1  2  3  4  5]
       #[ 6  7  8  9 10 11]
       #[ 0  0  0  0  0  0]]
       ```

       ```python
       #会生成一个新的张量b，填充a中原来的元素，但是a没有发生变化
       a=np.array([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11]])
       b=np.resize(a,(3,6))
       print(b)
       #[[ 0  1  2  3  4  5]
       #[ 6  7  8  9 10 11]
       #[ 0  1  2  3  4  5]]
       ```

- np.sum()

    > ```text
    > sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue)
    > ```

    1. sum()输入参数带有axis时，将按照指定axis进行对应求和

    ```python
    a=np.array([[ 0,  1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10, 11]])
    print("np.sum(a)",np.sum(a))
    print("np.sum(a)",np.sum(a,axis=0))
    print("np.sum(a)",np.sum(a,axis=1))
    
    #np.sum(a) 66
    #np.sum(a,axis=0) [ 6  8 10 12 14 16]
    #np.sum(a,axis=1) [15 51]
    ```

    2. sum()输入参数axis为多个轴时，则依次按要求在axis上进行多次求和

    ```python
    b=np.arange(12).reshape(2,2,3)
    b=np.array([
                [[ 0 , 1,  2],
                 [ 3 , 4 , 5]],
                [[ 6 , 7 , 8],
                 [ 9 , 10 ,11]]
                                ])
    print("np.sum(b,axis=(1,2)",np.sum(b,axis=(1,2)))
    print("np.sum(b,axis=(0,1)",np.sum(b,axis=(0,1)))
    print("np.sum(b,axis=(0,1,2)",np.sum(b,axis=(0,1,2)))
    
    #np.sum(b,axis=(1,2) [15 51] （2,2,3） 对第二和第三维求和，留下第一维即2
    #np.sum(b,axis=(0,1) [18 22 26] 对第一和第二维求和，留下第三维即3
    #np.sum(b,axis=(0,1,2) 66
    ```

    3. dtype先将a转化后再相加

    4. keepdims =True 将保持a的维度

       没有使用keepdims 时有维度丢失的可能，在代码中可能出现bug.

- Normalizing

    一种常用的在机器学习和深度学习中使用的技巧就是归一化，这会使得我们的梯度下降表现更佳，这意味着
    $$
    x →\frac{x}{\| x\|}
    $$
    For example, if 
    $$
    x = 
    \begin{bmatrix}
        0 & 3 & 4 \\
        2 & 6 & 4 \\
    \end{bmatrix}\tag{3}
    $$
    then
    $$
    \| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
        5 \\
        \sqrt{56} \\
    \end{bmatrix}\tag{4} 
    $$
    and        
    $$
    x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
        0 & \frac{3}{5} & \frac{4}{5} \\
        \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
    \end{bmatrix}\tag{5}
    $$
	```python
	
	# GRADED FUNCTION: normalizeRows
    
    def normalizeRows(x):
        """
        Implement a function that normalizes each row of the matrix x (to have unit length).
    
        Argument:
        x -- A numpy matrix of shape (n, m)
    
        Returns:
        x -- The normalized (by row) numpy matrix. You are allowed to modify x.
        """
    
        ### START CODE HERE ### (≈ 2 lines of code)
        # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
        x_norm=np.linalg.norm(x,axis=1,keepdims=True)
        # Divide x by its norm.
        x=x/x_norm
        ### END CODE HERE ###
    
        return x
    x = np.array([
        [0, 3, 4],
        [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalizeRows(x)))
	```

# 2. 向量化

我们在机器学习和深度学习时，我们往往需要处理大数据集，因此计算能力成为制约我们算法的关键，为了使得我们的计算更加快速，我们引入了向量化。

```
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
```

- dot

  ```python
  #normal
  for i in range(len(x1)):
      dot+= x1[i]*x2[i]
      
  #vectorization
  dot = np.dot(x1,x2)
  
  #dot = 278 向量乘法
  ```

- outer

  ```python
  #normal
  for i in range(len(x1)):
      for j in range(len(x2)):
          outer[i,j] = x1[i]*x2[j]
  #vectorization
  outer = np.outer(x1,x2)
  # 全连接 9*9的连接矩阵
  ```

- multiply

  ```python
  #normal
  for i in range(len(x1)):
      mul[i] = x1[i]*x2[i]
  #vectorization
  mul = np.multiply(x1,x2)
  #对应位置相乘，形成一行向量
  ```

# numpy-API

[the official documentation](https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.exp.html).