---
layout: single
title:  "데이터 시각화: Matplotlib"
categories: "Python"
tags: [Python]
toc: true
author_profile: false
sidebar:
  nav: "docs"
# search: false
---

# Python - 데이터 시각화: Matplotlib

기본적인 설명은 Wikidocs를 참고함<br>
실무에서 자주 쓰는 내용을 복습 용도로 정리하려고 함<br>
출처: https://wikidocs.net/124976<br>
데이터: Dacon 따릉이 예측 대회 훈련 데이터

### 라이브러리 설치


```python
# pip install matplotlib
```

### 라이브러리 Import 및 데이터 로드


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./dacon_ddarung_train.csv', index_col='id')
data = data.dropna() # 결측치 제거
data
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>16.3</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>89.0</td>
      <td>576.0</td>
      <td>0.027</td>
      <td>76.0</td>
      <td>33.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13</td>
      <td>20.1</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>48.0</td>
      <td>916.0</td>
      <td>0.042</td>
      <td>73.0</td>
      <td>40.0</td>
      <td>159.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>79.0</td>
      <td>1382.0</td>
      <td>0.033</td>
      <td>32.0</td>
      <td>19.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>23</td>
      <td>8.1</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>54.0</td>
      <td>946.0</td>
      <td>0.040</td>
      <td>75.0</td>
      <td>64.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>18</td>
      <td>29.5</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>7.0</td>
      <td>2000.0</td>
      <td>0.057</td>
      <td>27.0</td>
      <td>11.0</td>
      <td>431.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2174</th>
      <td>4</td>
      <td>16.8</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>53.0</td>
      <td>2000.0</td>
      <td>0.031</td>
      <td>37.0</td>
      <td>27.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>2175</th>
      <td>3</td>
      <td>10.8</td>
      <td>0.0</td>
      <td>3.8</td>
      <td>45.0</td>
      <td>2000.0</td>
      <td>0.039</td>
      <td>34.0</td>
      <td>19.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2176</th>
      <td>5</td>
      <td>18.3</td>
      <td>0.0</td>
      <td>1.9</td>
      <td>54.0</td>
      <td>2000.0</td>
      <td>0.009</td>
      <td>30.0</td>
      <td>21.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>2178</th>
      <td>21</td>
      <td>20.7</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>37.0</td>
      <td>1395.0</td>
      <td>0.082</td>
      <td>71.0</td>
      <td>36.0</td>
      <td>216.0</td>
    </tr>
    <tr>
      <th>2179</th>
      <td>17</td>
      <td>21.1</td>
      <td>0.0</td>
      <td>3.1</td>
      <td>47.0</td>
      <td>1973.0</td>
      <td>0.046</td>
      <td>38.0</td>
      <td>17.0</td>
      <td>170.0</td>
    </tr>
  </tbody>
</table>
<p>1328 rows × 10 columns</p>
</div>



### Line Chart 그리기


```python
Y = data['hour'].values
colors = np.random.rand(1328)

plt.figure(figsize=(7,4))
plt.plot(Y, c='r', linestyle='--',linewidth=1, alpha=0.9, label='Y')    # s is size, c is color , alpha=transparency, label=legend
plt.xlim(0,120)    # xlim: x Axis range
plt.ylim(1,21)    # xlim: x Axis range
plt.title("Dacon DDarung Y: Count")
plt.grid()
plt.legend(loc='upper right')    # loc is location
plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img1.png" alt="img1" style="zoom:67%;" />    


### Scatter Plot 그리기


```python
X = data['hour'].values
Y = data['count'].values
Z = data['hour_bef_windspeed'].values

plt.figure(figsize=(7,7))
plt.scatter(X, Y, s=2, c='r', alpha=0.4) # s is size, c is color , alpha=transparency
plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img2.png" alt="img2" style="zoom:67%;" />      


#### Hexbin


```python
plt.figure(figsize=(7,7))
plt.hexbin(X, Y)
plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img3.png" alt="img3" style="zoom:67%;" />          


### 3D Scatter Plot


```python
from mpl_toolkits.mplot3d import Axes3D

X = data['hour'].values
Y = data['count'].values
Z = data['hour_bef_windspeed'].values

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='green', marker='o', s=15)

plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img4.png" alt="img4" style="zoom:67%;" />     


### Bar Chart


```python
x = np.arange(3)                     # 막대 그래프의 X축 높이
ticks = list(np.arange(1, 4))        # 막대 그래프의 X축 눈금, 위 x와 길이가 같아야 함.
y_values = [Y[i] for i in range(3)] # 막대 그래프 별 높이, 위 x와 길이가 같아야 함.

plt.xticks(x, ticks)
plt.bar(x, y_values, width=0.5, align='center', # Width: 막대 그래프의 너비, align: X축 눈금의 위치,
        edgecolor='black', linewidth=5      # edge: 테두리 색, linewidth: 테두리 굵기
       )                  
plt.show()
```

<img src="/assets/images/2022-02-09-Matplotlib/img5.png" alt="img5" style="zoom:67%;" />      


### 수평, 수직선 표기 및 영역 강조


```python
x = np.arange(0, 4, 0.5)

plt.plot(x, x + 1, 'bo')
plt.plot(x, x**2 - 4, 'g--')
plt.plot(x, -2*x + 3, 'r:')

plt.axhline(4.0, 0.1, 0.9, color='lightgray', linestyle='--', linewidth=2)
plt.hlines(-0.62, 1.0, 2.5, color='gray', linestyle='solid', linewidth=3)
plt.axvline(1.0, 0.2, 0.8, color='lightgray', linestyle='--', linewidth=2)
plt.vlines(1.8, -3.0, 2.0, color='gray', linestyle='solid', linewidth=3)

plt.show()
```

<img src="/assets/images/2022-02-09-Matplotlib/img6.png" alt="img6" style="zoom:67%;" />   

```python
x = np.arange(0, 4, 0.5)

plt.plot(x, x + 1, 'bo')
plt.plot(x, x**2 - 4, 'g--')
plt.plot(x, -2*x + 3, 'r:')

plt.axvspan(xmin=2, xmax=3, color='g', alpha=0.5)
plt.axhspan(ymin=2, ymax=3, color='gray', alpha=0.5)
plt.show()
```

<img src="/assets/images/2022-02-09-Matplotlib/img7.png" alt="img7" style="zoom:67%;" />          


### 객체 지향 인터페이스


```python
fig, ax = plt.subplots(2, 2, figsize=(7,7),)     # rows num , cols num  , the size of canvas
plt.show();

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True) # sharex : share the X axis
plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img8.png" alt="img8" style="zoom:67%;" />  
    

<img src="/assets/images/2022-02-09-Matplotlib/img9.png" alt="img9" style="zoom:67%;" />  
    


#### 여러 개의 그래프 그리기


```python
X = data['hour'].values[:10]
Y = data['count'].values[:10]
Z = data['hour_bef_windspeed'].values[:10]
W = data['hour_bef_temperature'].values[:10]

plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
plt.plot(X, c='k', label='hour')
plt.legend(loc='upper right')

plt.subplot(2,2,2)
plt.plot(Y, c='b', label='count')
plt.legend(loc='upper right')

plt.subplot(2,2,3)
plt.plot(Z, c='r', label='hour_bef_windspeed')
plt.legend(loc='upper right')

plt.subplot(2,2,4)
plt.plot(W, c='g', label='hour_bef_temperature')
plt.legend(loc='upper right')

plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img10.png" alt="img10" style="zoom:67%;" />      


#### 여러 그래프 간의 축 공유


```python
X = data['hour'].values[:10]
Y = data['count'].values[:10]
Z = data['hour_bef_windspeed'].values[:10]
W = data['hour_bef_temperature'].values[:10]

fig, ax = plt.subplots(2, 2, figsize=(12,7), sharex=True, sharey=True)    # Y축 공유를 하게 되면 Scale에 따라 그래프를 구분하기 힘듦
ax[0][0].plot(X, c='k', label='hour')
ax[0][0].set_title('LineChart1')
ax[0][0].legend(loc='upper right')

ax[0][1].plot(Y, c='b', label='count')
ax[0][1].set_title('LineChart2')
ax[0][1].legend(loc='upper left')

ax[1][0].plot(Z, c='r', label='hour_bef_windspeed')
ax[1][0].set_title('LineChart3')
ax[1][0].legend(loc='lower left')

ax[1][1].plot(W, c='g', label='hour_bef_temperature')
ax[1][1].set_title('LineChart4')
ax[1][1].legend(loc='upper center')

plt.show();

fig, ax = plt.subplots(2, 2, figsize=(12,7), sharex=True)    # Y축 공유를 하게 되면 Scale에 따라 그래프를 구분하기 힘듦
ax[0][0].plot(X, c='k', label='hour')
ax[0][0].set_title('LineChart1')
ax[0][0].legend(loc='upper right')

ax[0][1].plot(Y, c='b', label='count')
ax[0][1].set_title('LineChart2')
ax[0][1].legend(loc='upper left')

ax[1][0].plot(Z, c='r', label='hour_bef_windspeed')
ax[1][0].set_title('LineChart3')
ax[1][0].legend(loc='lower left')

ax[1][1].plot(W, c='g', label='hour_bef_temperature')
ax[1][1].set_title('LineChart4')
ax[1][1].legend(loc='upper center')

plt.show();
```


 <img src="/assets/images/2022-02-09-Matplotlib/img11.png" alt="img11" style="zoom:67%;" />  

<img src="/assets/images/2022-02-09-Matplotlib/img12.png" alt="img12" style="zoom:67%;" />  


#### Y축 추가하기


```python
X = data['hour'].values[:10]
Y = data['count'].values[:10]

fig, ax1 = plt.subplots(figsize=(12,4))
ax1.set_xlabel('Index')
ax1.set_ylabel('Hour')
line1 = ax1.plot(X, c='k', label='hour')

ax2 = ax1.twinx()
ax2.set_ylabel('count')
line2 = ax2.plot(Y, c='b', label='count')

lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.show();
```

<img src="/assets/images/2022-02-09-Matplotlib/img13.png" alt="img13" style="zoom:67%;" />  


### 도표 내에 화살표 표기


```python
plt.tight_layout()
plt.axis([0,100,0,100])
plt.plot(20, 20, "bs")
plt.plot(50, 50, "rs")
plt.plot(80, 80, "bs")
plt.annotate('Point', xy=(50, 50), xytext=(50,20),
            fontsize=14, ha='center',
            arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=15))
plt.show()
```

<img src="/assets/images/2022-02-09-Matplotlib/img14.png" alt="img14" style="zoom:67%;" />      


### 참고: 이미지 저장하기


```python
X = data['hour'].values[:10]
Y = data['count'].values[:10]

fig, ax1 = plt.subplots(figsize=(12,4))
ax1.set_xlabel('Index')
ax1.set_ylabel('Hour')
line1 = ax1.plot(X, c='k', label='hour')

ax2 = ax1.twinx()
ax2.set_ylabel('count')
line2 = ax2.plot(Y, c='b', label='count')

lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.savefig('save_plt_img.png', dpi=150, facecolor='#d49f9f', # dpi : 해상도, facecolor: 배경 색
            edgecolor='red', bbox_inches='tight')         # edgecolor: 테두리 색깔, bbox_inches:  그래프로 저장할 부분임
```

<img src="/assets/images/2022-02-09-Matplotlib/img15.png" alt="img15" style="zoom:67%;" />      



```python
from IPython.display import Image
Image(filename='save_plt_img.png') 
```

<img src="/assets/images/2022-02-09-Matplotlib/img16.png" alt="img16" style="zoom:67%;" />      



### 참고: matplotlib.pyplot 공통 서식 통일 시키기


```python
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12
```

### 참고: Matplotlib 한글 폰트 적용하기
<pre>
출처: https://teddylee777.github.io/visualization/matplotlib-%EC%8B%9C%EA%B0%81%ED%99%94-%ED%95%9C%EA%B8%80%ED%8F%B0%ED%8A%B8%EC%A0%81%EC%9A%A9
나눔고딕 설치: https://hangeul.naver.com/font
나눔고딕 폰트를 설치해야 아래의 font에 나눔고딕이 적용가능함.
</pre>


```python
# unicode minus를 사용하지 않기 위한 설정 (minus 깨짐현상 방지)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'NanumGothic'
```


```python
X = data['hour'].values[:10]
Y = data['count'].values[:10]

fig, ax1 = plt.subplots(figsize=(12,4))
ax1.set_xlabel('Index')
ax1.set_ylabel('Hour')
line1 = ax1.plot(X, c='k', label='시간')

ax2 = ax1.twinx()
ax2.set_ylabel('count')
line2 = ax2.plot(Y, c='b', label='따릉이 대여 수')

lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.savefig('save_plt_img.png', dpi=150, facecolor='#d49f9f', # dpi : 해상도, facecolor: 배경 색
            edgecolor='red', bbox_inches='tight')             # edgecolor: 테두리 색깔, bbox_inches:  그래프로 저장할 부분임
```

<img src="/assets/images/2022-02-09-Matplotlib/img17.png" alt="img17" style="zoom:67%;" />      

