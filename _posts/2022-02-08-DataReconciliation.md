---
layout: single
title:  "데이터 보정 이해 및 실습 코드"
categories: "Data Analysis"
tags: [Python, Data Analysis, Data Reconciliation]
toc: true
author_profile: false
sidebar:
  nav: "docs"
# search: false
---

# 데이터 보정: Data Reconciliation

## 데이터 보정이란,

<pre>
공장의 단위 공정에서 수많은 변수 데이터가 수집될 때, 다양한 측정 오차가 포함되어 있음.
공정 분석은 보통 물질 수지와 열 수지를 만족하는 것으로 시작되는데 
측정 오차가 포함된 상태에서는 이들을 만족시키기 어려움.
이러한 문제를 해결하기 위한 방법으로 <big><strong><ins>데이터 보정(Data Reconciliation; DR)</ins></strong></big> 기법을 사용함.

DR이란 측정값의 통계적 확률 분포를 이용하여 
물질 수지와 열 수지를 만족하도록 측정값의 오차를 소거하는 기술임.
DR은 측정값 내 노이즈의 종류에 따라 <strong> White Noise 보정, Gross Error 보정</strong>으로 나눌 수 있다.
</pre>



<pre>측정값 내 노이즈의 종류: White Noise(좌), Gross Error(우)</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img1.png" alt="DR_img1" style="zoom:67%;" />

## 데이터 보정의 종류 - White Noise 보정

<pre>
무작위 형태의 오차(White Noise, Random Error)가 있는 경우의 DR기법은 비선형 최적화 문제를 구성함.
목적함수는 다음의 식과 같이 최소제곱법(Least Square)에 의한 2차식으로 표현가능함.
White Noise 보정의 DR 기법은 최적화 문제의 해, 보정값을 구함.
</pre>

<img src="/assets/images/2022-02-08-DataReconciliation/DR_img2.png" alt="DR_img2" style="zoom:67%;"/>

<pre>
무작위 형태의 오차(White Noise, Random Error)가 있는 경우의 DR기법은 비선형 최적화 문제를 구성함.
목적함수는 다음의 식과 같이 최소제곱법(Least Square)에 의한 2차식으로 표현가능함.
White Noise 보정의 DR 기법은 최적화 문제의 해, 보정값을 구함.
</pre>

## 데이터 보정의 종류 - Gross Error 보정

<pre>
측정값에 편향 오차가 포함되어 있다면 White Noise 보정에 별도의 과정이 필요함. 
Gross Error(편향 오차)를 제거하지 않고 White Noise 보정을 실시한다면 
갖고 있는 Gross Error(편향 오차)는 보정되겠지만 다른 변수의 측정값에 오차를 떠넘기게 되어 
정상적인 측정값을 갖는 변수의 입장에서는 데이터 보정을 실행하지 않는 것보다 못하게 됨.

### Gross Error 보정 단계

a) Gross Error(편향 오차) 보정을 하기 위해서는 Gross Error(편향 오차)가 있는지 확인한 후, 

b) Gross Error(편향 오차)를 갖는 변수를 찾아내어, 

c) Gross Error(편향 오차)를 제거하여 데이터 보정을 실시해야 함.

이를 아래처럼 표현할 수 있다. 

a) Global Test 실시 : Fault Detection 단계 진행

b) Fault Identification 단계 진행

c) Reconstruction 단계 진행
</pre>

<pre>
Gross Error(편향 오차)를 제거하지 않고 데이터 보정을 실시한 사례
</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img3_4.png" alt="DR_img3_4" style="zoom:67%;" />

<pre>
Gross Error 보정의 순서도
</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img5.png" alt="DR_img5" style="zoom:67%;" />

<pre>
Gross Error 보정 중 Global Test 공식
</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img6.png" alt="DR_img6" style="zoom:67%;" />

### Gross Error 보정 -1. Global Test

<pre>측정값 과 보정값의 차이가 큰 변수에 편향 오차가 존재할 가능성이 크기에 
Global Test의 r(감마 혹은 검정치)가 커질 것임.
r(감마 혹은 검정치)를 판단하는 기준치로는 카이-제곱(χ2) 통계량을 이용함.
아래의 식과 같이 r(감마 혹은 검정치)가 카이-제곱(χ2) 기준치보다 크면 
Gross Error(편향 오차)가 존재한다고 판단 </pre>


<pre>
Gross Error 보정 중 Global Test 공식-기준치 판단
</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img7.png" alt="DR_img7" style="zoom:67%;" />

<pre>측정값 과 보정값의 차이가 큰 변수에 편향 오차가 존재할 가능성이 크기에 
Global Test의 r(감마 혹은 검정치)가 커질 것임.
r(감마 혹은 검정치)를 판단하는 기준치로는 카이-제곱(χ2) 통계량을 이용함.
아래의 식과 같이 r(감마 혹은 검정치)가 카이-제곱(χ2) 기준치보다 크면 
<u>Gross Error(편향 오차)가 존재한다고 판단</u>strong> 
</pre>

### Gross Error 보정 -2. Fault Identification & Reconstruction

<pre> 다음의 3가지 단계 중 1) 을 통해 Gross Error(편향 오차)의 존재를 판단함. 
     1) Global Test 실시 : Fault Detection 단계 진행
     2) Fault Identification 단계 진행
     3) Reconstruction 단계 진행
이제 어느 변수가 Gross Error를 갖고 있는지 확인하는 Identification(식별) 임. 
Identification(식별) 단계에서는 여러가지 알고리즘이 있겠지만, 
이 논문(출처 참고)에서는 순서에 따른 시행착오법을 이용함. 
측정값과 보정값의 차이가 가장 큰 측정 변수부터 Global Test의 검정치 산출 공식에서 
제외하여 검정치를 산출함. 즉, 검정치를 한번 더 계산한다. 한 개의 변수를 제외한 채로. 
만약 재계산된 검정치가 기준치보다 낮아졌다면 
<u>해당 변수에서 기준치를 초과하게 만드는 원인이 있다고 판단하는 것임.</u>
이 과정을 측정값과 보정값의 차이의 순서대로 변수 전체에 반복함.
 </pre>

### Gross Error 보정 사례-Excel 해찾기


<pre>Excel 해찾기를 이용하한 Gross Error 보정 사례-Gross Error(편향 오차)식별</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img8.png" alt="DR_img8" style="zoom:67%;" />

<pre>측정값 과 보정값의 차이가 큰 변수에 편향 오차가 존재할 가능성이 크기에 
Global Test의 r(감마 혹은 검정치)가 커질 것임.
r(감마 혹은 검정치)를 판단하는 기준치로는 카이-제곱(χ2) 통계량을 이용함.
아래의 식과 같이 r(감마 혹은 검정치)가 카이-제곱(χ2) 기준치보다 크면 Gross Error(편향 오차)가 존재한다고 판단 </pre>


<pre>Excel 해찾기를 이용하한 Gross Error 보정 사례-Gross Error(편향 오차) 변수 판별</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img9.png" alt="DR_img9" style="zoom:67%;" />

<pre>Excel 해찾기를 이용하한 Gross Error 보정 사례- 결과</pre>
<img src="/assets/images/2022-02-08-DataReconciliation/DR_img10.png" alt="DR_img10" style="zoom:67%;" />

### 마치며.

<pre>단, 내가 이해한 바로는 이 방법은 Gross Error가 1개 있다고 가정한 것임. 
2개이상 편향오차가 발생하지 않는다고 가정한 것임. 
공정 내에 다양한 측정 센서들이 동시에 고장나지 않는다라는 현장 엔지니어들의 소견도 그랬음.

다만 아쉬운 것은 이 논문(출처 참고)에는 1개 일 때와 2개 이상 일 때를 구분했으나 2개 이상일 때에 대해서 실습이라던가, 
별도의 공부를 아직 까지 진행하지 못함.

<u>데이터 보정을 통해서 공장 내 존재하는 노이즈를 보정할 수 있다면 데이터 분석 및 공정 모니터링에 많은 도움이 될 것으로 기대함.</u>

 <ul>
참고 문헌:
Martini, A, Coco, D, Sorce, A, Traverso, A, & Levorato, P. "Gross Error Detection Based on Serial Elimination: Applications to an Industrial Gas Turbine." Proceedings of the ASME Turbo Expo 2014: Turbine Technical Conference and Exposition. Volume 3A: Coal, Biomass and Alternative Fuels; Cycle Innovations; Electric Power; Industrial and Cogeneration. Düsseldorf, Germany. June 16–20, 2014. V03AT07A024. ASME. https://doi.org/10.1115/GT2014-26746
Data Processing and Reconciliation for Chemical Process Operations
1st Edition - October 11, 1999
Authors: José Romagnoli, Mabel Sanchez
eBook ISBN: 9780080530277
Miao, Yu & Su, Hongye & Rong, Gang & Chu, Jian. (2009). Industrial Processes: Data Reconciliation and Gross Error Detection. Measurement and Control. 42. 209-215. 10.1177/002029400904200704. 
</ul>
</pre>

### 구현

이를 코드로 만들면 아래와 같음.<br>
사용라이브러리는 최적화 라이브러리인 Gekko를 이용하였음.


```python
!pip install gekko
```

    Collecting gekko
      Downloading gekko-1.0.2-py3-none-any.whl (12.4 MB)
    Requirement already satisfied: numpy>=1.8 in c:\users\asdm1\anaconda3\envs\techdas\lib\site-packages (from gekko) (1.21.5)
    Installing collected packages: gekko
    Successfully installed gekko-1.0.2



```python
import numpy as np
from gekko import GEKKO
from scipy.stats import chi2


class DR:
    """
        - Data Reconciliation
        - Gross Error Detection
        - Gross Error Identification
        - Return reconciliation of x
        * Example
            dr = DR()
            x_recon = mydr.reconciliate(jacobian, measured_x, sigma_from_data)
    """

    def __init__(self):
        self.show_result = True

    def _initialize_x(self, m, text, lb_=0, ub_=10000):
        for i in range(self.xnum):
            globals()[text + '{}'.format(i + 1)] = m.Var(lb=lb_, ub=ub_)
            globals()[text + '{}'.format(i + 1)].value = self.measured_x[i]

    def _set_consts(self, m, text, eq):
        for eqs in range(self.eqnum):
            m.Equation(sum([globals()[text + '{}'.format(i + 1)] *
                            self.jacobian[eqs][0 + i] for i in range(self.xnum)]) == eq)

    def _copy_x(self, text):
        copied_x = []
        for i in range(self.xnum):
            copied_x.append(globals()[text + '{}'.format(i + 1)])
        copied_x = np.array(copied_x)
        return copied_x

    def _solve(self, m, obj_func, show=True):
        m.Obj(obj_func)
        m.options.IMODE = 3  # steady state optimization
        if show:
            m.solve(disp=self.show_result)
        else:
            m.solve(disp=show)

    def _calculate_gamma(self, xs1, xs2):
        gamma = 0
        for i in range(self.xnum):
            gamma += ((xs1[i] - xs2[i]) / self.sigma_from_data[i]) ** 2
        gamma = float(gamma)
        return gamma

    def reconciliate(self, jacobian, measured_x, sigma_from_data):
        """
        :parameter
            jacobian: list
            measured_x: list
            sigma_from_data: Numpy Array
        :return:
            reconcil_x = Numpy Array
        """
        self.xnum = len(measured_x)
        self.eqnum = len(jacobian)

        self.jacobian = jacobian
        self.measured_x = measured_x
        # 예제를 위한 값으로 설정값으로 지정
        self.sigma_from_data = np.array(sigma_from_data)
        self.chi_square_1 = chi2.ppf(0.95, self.xnum - 1)
        self.chi_square_2 = chi2.ppf(0.95, self.xnum - 2)

        # Initialize Model
        m = GEKKO()
        text1 = 'x'

        # define parameter
        eq = m.Param(value=0)

        # initialize variables, initial values
        self._initialize_x(m, text1)

        # Equations
        self._set_consts(m, text1, eq)

        # Define the objective function
        obj_func = sum(
            [((globals()['x{}'.format(i + 1)] - self.measured_x[i]) / self.sigma_from_data[i]) ** 2 for i in
             range(self.xnum)])

        # Optimize
        self._solve(m, obj_func)
        reconcil_x = self._copy_x(text1)

        if self.show_result:
            print('Results of Fault Detect')
            for i in range(self.xnum):
                print(f'x{i + 1}:\t{round(float(reconcil_x[i]),3)}')

        # Detection
        gamma = self._calculate_gamma(reconcil_x, self.measured_x)
        if gamma > self.chi_square_1:
            print(
                f"Gross Error 검정치({round(gamma, 2)}) > 카이제곱 기준치({round(self.chi_square_1, 2)}): Gross Error 존재.")
            self.identify()
            reconcil_x = self.reconil_x2

        return reconcil_x

    def identify(self):
        text2 = 'id'
        for k in range(self.xnum):
            # Initialize Model
            m2 = GEKKO()
            eq2 = m2.Param(value=0)

            # initialize variables, initial values
            self._initialize_x(m2, text2)

            # Equations
            self._set_consts(m2, text2, eq2)

            # Define the objective function
            obj_func2 = sum(
                [((globals()['id{}'.format(i + 1)] - self.measured_x[i]) / self.sigma_from_data[i]) ** 2 for i in
                 range(self.xnum)])
            term = ((globals()['id{}'.format(k + 1)] - self.measured_x[k]) / self.sigma_from_data[k]) ** 2

            # Optimize
            self._solve(m2, obj_func2 - term, show=False)
            reconcil_x2 = self._copy_x(text2)

            # Identification
            term_value = (
                                 (reconcil_x2[k] - self.measured_x[k]) / self.sigma_from_data[k]) ** 2
            gamma_id = self._calculate_gamma(reconcil_x2, self.measured_x)
            gamma_id = float(gamma_id - term_value)
            print(f"[{k + 1}번째 변수 식별 중] 검정치:\t{gamma_id}")

            if gamma_id < self.chi_square_2:
                print(f"x{k + 1}번째가 Gross Error를 갖고 있음")

                if self.show_result:
                    print('Results of Fault Identification')
                    for i in range(self.xnum):
                        print(f'x{i + 1}:\t{round(float(reconcil_x2[i]),3)}')
                    print(reconcil_x2)
                    self.reconil_x2 = reconcil_x2
                    break
```

참고자료: https://github.com/Kyuhan1230/DataReconciliation.git - DR_Example_v1.xlsm


```python
dr = DR()
# 참고자료 내 Jacobian 행렬
jacobian = [[1, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1]]
# 참고자료 내 계측값(현재값)
measured_x = [100, 150, 80, 250, 330.0, 250, 110, 100, 80, 60, 40, 40, 50, 60, 90, 150]
# 참고자료 내 표준편차
sigma = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```


```python
'''아래는 참고자료 내 변수 별 계측값의 정답(오차가 없다면)을 나타낸 것임.
즉, 나는 X6에 Gross Error를 삽입함.'''
answer_x = [100, 150, 80, 250, 330.0, 120, 110, 100, 80, 60, 40, 40, 50, 60, 90, 150]
```


```python
# Data Reconciliation 결과:
# Fault Detection > Identify 까지 진행된 결과가 출력됨.
x_recon = dr.reconciliate(jacobian, measured_x, sigma_from_data=sigma)
```

    apm 58.141.79.44_gk_model0 <br><pre> ----------------------------------------------------------------
     APMonitor, Version 1.0.1
     APMonitor Optimization Suite
     ----------------------------------------------------------------


​     
​     --------- APM Model Size ------------
​     Each time step contains
​       Objects      :            0
​       Constants    :            0
​       Variables    :           17
​       Intermediates:            0
​       Connections  :            0
​       Equations    :            9
​       Residuals    :            9
​     
     Number of state variables:             16
     Number of total equations: -            8
     Number of slack variables: -            0
     ---------------------------------------
     Degrees of freedom       :              8
     
     **********************************************
     Steady State Optimization with Interior Point Solver
     **********************************************


​      
​     Info: Exact Hessian
​    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    
    This is Ipopt version 3.12.10, running with linear solver ma57.
    
    Number of nonzeros in equality constraint Jacobian...:      128
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:       16
    
    Total number of variables............................:       16
                         variables with only lower bounds:        0
                    variables with lower and upper bounds:       16
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        8
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  0.0000000e+00 1.30e+02 0.00e+00   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  3.4819973e+02 5.68e-14 8.78e-03  -0.1 6.65e-01    -  9.92e-01 1.00e+00h  1
       2  3.4819428e+02 5.68e-14 1.15e-02  -5.9 3.34e-03    -  9.90e-01 1.00e+00f  1
       3  3.4819428e+02 1.14e-13 1.53e-04  -7.9 2.15e-05    -  9.90e-01 1.00e+00h  1
       4  3.4819428e+02 5.68e-14 5.45e-08  -9.9 2.19e-09    -  1.00e+00 1.00e+00h  1
    
    Number of Iterations....: 4
    
                                       (scaled)                 (unscaled)
    Objective...............:   3.4819428123775947e+02    3.4819428123775947e+02
    Dual infeasibility......:   5.4472337173709296e-08    5.4472337173709296e-08
    Constraint violation....:   3.7895612573872014e-14    5.6843418860808015e-14
    Complementarity.........:   1.2572229096863036e-07    1.2572229096863036e-07
    Overall NLP error.......:   1.2572229096863036e-07    1.2572229096863036e-07


​    
​    Number of objective function evaluations             = 5
​    Number of objective gradient evaluations             = 5
​    Number of equality constraint evaluations            = 5
​    Number of inequality constraint evaluations          = 0
​    Number of equality constraint Jacobian evaluations   = 5
​    Number of inequality constraint Jacobian evaluations = 0
​    Number of Lagrangian Hessian evaluations             = 4
​    Total CPU secs in IPOPT (w/o function evaluations)   =      0.004
​    Total CPU secs in NLP function evaluations           =      0.002
​    
    EXIT: Optimal Solution Found.
     
     The solution was found.
     
     The final value of the objective function is    348.194281237759     
     
     ---------------------------------------------------
     Solver         :  IPOPT (v3.12)
     Solution time  :   1.040000000011787E-002 sec
     Objective      :    348.194281237759     
     Successful solution
     ---------------------------------------------------
     
    Results of Fault Detect
    x1:	103.819
    x2:	153.819
    x3:	91.457
    x4:	257.638
    x5:	349.095
    x6:	183.04
    x7:	86.373
    x8:	79.683
    x9:	116.408
    x10:	53.075
    x11:	29.765
    x12:	66.631
    x13:	33.298
    x14:	49.918
    x15:	99.929
    x16:	149.847
    Gross Error 검정치(348.19) > 카이제곱 기준치(25.0): Gross Error 존재.
    [1번째 변수 식별 중] 검정치:	346.6666666949334
    [2번째 변수 식별 중] 검정치:	346.6666666949334
    [3번째 변수 식별 중] 검정치:	336.1827956955419
    [4번째 변수 식별 중] 검정치:	343.7713436946043
    [5번째 변수 식별 중] 검정치:	321.5593705549557
    [6번째 변수 식별 중] 검정치:	0.0
    x6번째가 Gross Error를 갖고 있음
    Results of Fault Identification
    x1:	100.0
    x2:	150.0
    x3:	80.0
    x4:	250.0
    x5:	330.0
    x6:	120.0
    x7:	110.0
    x8:	100.0
    x9:	80.0
    x10:	60.0
    x11:	40.0
    x12:	40.0
    x13:	50.0
    x14:	60.0
    x15:	90.0
    x16:	150.0
    [[100.        ]
     [150.        ]
     [ 80.00000001]
     [250.        ]
     [330.00000001]
     [120.00000001]
     [110.        ]
     [100.        ]
     [ 79.99999999]
     [ 60.        ]
     [ 40.00000001]
     [ 40.00000001]
     [ 50.        ]
     [ 59.99999999]
     [ 90.00000001]
     [150.        ]]

<pre>아래의 순서를 모두 만족한 후, 데이터 보정값을 찾아내었음.
 1) Gross Error(편향 오차) 보정을 하기 위해서는 Gross Error(편향 오차)가 있는지 확인한 후, 
 2) Gross Error(편향 오차)를 갖는 변수를 찾아내어, 
 3) Gross Error(편향 오차)를 제거하여 데이터 보정을 실시해야 함.
 현재는 X6에만 Gross Error가 있고 
 다른 값에는 어떠한 Error, Noise가 들어가 있지 않아 보정된 데이터와 정답이 매우 유사하나 
 실제 데이터는 이와 사뭇 다름.
 </pre>

