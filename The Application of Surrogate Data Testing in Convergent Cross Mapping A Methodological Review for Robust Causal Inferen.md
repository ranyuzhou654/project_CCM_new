# The Application of Surrogate Data Testing in Convergent Cross Mapping: A Methodological Review for Robust Causal Inference



**Abstract**

This report provides a comprehensive review of the integration of surrogate data testing with Convergent Cross Mapping (CCM) for causal inference in nonlinear dynamical systems. We begin by elucidating the theoretical foundations of CCM, rooted in Takens' theorem, and the fundamental principles of surrogate data as a non-parametric method for hypothesis testing. We then present a detailed taxonomy of surrogate generation algorithms—from simple random permutations to sophisticated methods like Iterated Amplitude Adjusted Fourier Transform (IAAFT) and Twin Surrogates—analyzing the specific null hypotheses each method is designed to test. The core of the report focuses on the practical application of these surrogates to assess the statistical significance of CCM cross-map skill and, critically, to detect and mitigate spurious causality arising from synchrony, common drivers, and other confounding factors. Through a synthesis of the scientific literature, we establish best practices, highlight common pitfalls, and discuss the profound implications of choosing an appropriate surrogate method. We conclude that surrogate data testing is not merely an optional add-on but an indispensable component for achieving robust and credible causal inference with CCM.



## Introduction



The fundamental scientific challenge of distinguishing true causal relationships from mere correlation in observational time series is particularly acute in complex systems. Fields ranging from ecology and climatology to neuroscience and finance are replete with systems characterized by nonlinear, nonseparable, and synergistic dynamics, where traditional linear methods for causal inference are often inadequate. Convergent Cross Mapping (CCM) has emerged as a powerful, equation-free technique derived from dynamical systems theory, designed specifically to detect causation in these challenging scenarios. Its core premise is that a causal variable leaves an informational "signature" on the affected variable, which can be detected through the skillful cross-prediction of system states.

However, any claim of causality, regardless of the sophistication of the detection method, requires rigorous statistical validation. The output of the CCM algorithm is a correlation coefficient, ρ, which quantifies the skill of cross-prediction. A high value of 

ρ is suggestive of a causal link, but it is not definitive proof. This raises a critical question: how high must this value be to be considered statistically significant? Without a formal statistical framework, any chosen threshold for ρ would be arbitrary. Furthermore, phenomena such as strong coupling, dynamic resonance, or the influence of a common external driver (e.g., seasonality) can induce synchrony between variables, leading to high ρ values that create the illusion of a direct causal link, often referred to as "spurious bidirectional causal relationships".

This imperative for statistical rigor introduces the method of surrogate data, a non-parametric, Monte Carlo-based approach for hypothesis testing. By generating an ensemble of artificial time series that adhere to a specific null hypothesis (e.g., "no causality"), this method provides a null distribution against which the observed result can be compared. This comparison allows for the calculation of a p-value, formally assessing whether the observed cross-map skill is a statistically significant finding or merely an artifact of the data's inherent properties or confounding factors. The use of surrogate data thus elevates the CCM analysis from a descriptive measure to a formal statistical test, making it an indispensable component of the methodology.

This report provides an exhaustive, expert-level synthesis of how, why, and which surrogate data methods are used in conjunction with CCM, based on a thorough review of the scientific literature. It will first detail the theoretical foundations of CCM, then provide a taxonomy of relevant surrogate generation techniques, and finally, detail their integrated application for robust causal inference.



## 1. Theoretical Foundations of Convergent Cross Mapping





### 1.1 State-Space Reconstruction and Takens' Theorem



The theoretical bedrock of CCM is the ability to reconstruct the multidimensional state space of a dynamical system from a single observed time series. A system's behavior over time can be visualized as a trajectory on a geometric object called a manifold, where each point on the manifold represents a unique state of the system. Takens' Embedding Theorem provides the mathematical justification that, under general conditions, a topologically equivalent "shadow manifold" can be reconstructed from a single time series of observations from that system.

This reconstruction is achieved through time-delay embedding. For a time series X, an E-dimensional state vector, or delay vector, is created for each point in time, Xt, by combining its present value with E−1 of its past values, each separated by a time lag τ. The resulting vector is x~t=(Xt,Xt−τ,Xt−2τ,…,Xt−(E−1)τ). The collection of all such vectors forms the shadow manifold, 

Mx. The choice of the embedding dimension E and the time lag τ are critical parameters, and methods exist for determining their optimal values to ensure a faithful reconstruction of the system's dynamics.



### 1.2 The Logic of Cross-Mapping



CCM leverages a key corollary to Takens' Theorem: if two variables, X and Y, are part of the same coupled dynamical system, their individually reconstructed shadow manifolds, Mx and My, are diffeomorphic to each other and to the true system manifold. This topological equivalence means there is a one-to-one mapping between points on the two manifolds.

The central premise of CCM is that if variable X causally influences variable Y, then the historical information of X is necessarily encoded in the dynamics of Y. Consequently, nearby points on the manifold reconstructed from 

Y (My) will correspond to nearby points on the manifold reconstructed from X (Mx). This property allows one to use the library of points on My to make skillful predictions of the state of X. This process is known as cross-mapping, denoted as X^∣My. This cross-prediction is typically asymmetric. If 

X unidirectionally forces Y, then Y will contain information about X, making the prediction X^∣My skillful. However, X will not contain information about Y, so the reverse prediction, Y^∣Mx, will not be skillful.



### 1.3 The Principle of Convergence



The defining feature of CCM that distinguishes it from simple correlation is the principle of convergence. The skill of the cross-map prediction, quantified by the Pearson correlation coefficient (ρ) between the predicted values (X^∣My) and the actual values (X), depends on the density of points used to construct the manifold.

The definitive test for causation in CCM is observing "convergence": as the library length L (the number of time points used to build the manifold) is increased, the reconstructed manifold becomes denser and more complete. If a causal link exists, this increased density allows for more accurate identification of nearest neighbors, and thus the cross-map prediction skill ρ should systematically increase and eventually saturate at a high value. This signature of convergence is considered a necessary condition for causation, as it demonstrates that the relationship is a fundamental property of the shared system dynamics rather than a coincidental correlation.



### 1.4 Distinctions from Granger Causality and Application Domains



CCM is fundamentally different from Granger Causality (GC). While GC is a statistical concept based on predictability and temporal precedence in stochastic, separable systems (where causal influences are independent), CCM is rooted in dynamical systems theory and is specifically designed for deterministic, nonseparable systems where variables may have synergistic effects. CCM can identify linkages between variables that may appear uncorrelated using standard linear methods.

The utility of CCM has been demonstrated across a wide array of scientific fields. Applications include identifying causal relationships in ecological systems (e.g., predator-prey dynamics, sardine populations and ocean temperatures), climatology (e.g., cosmic rays and global temperatures), finance (e.g., stock prices and trading volumes), neuroscience (e.g., brain connectivity), and hydrology (e.g., groundwater and streamflow).



## 2. The Method of Surrogate Data for Hypothesis Testing





### 2.1 Philosophy of the Method: Statistical Proof by Contradiction



Surrogate data testing is a non-parametric, Monte Carlo technique that functions as a form of statistical proof by contradiction. The core logic is to test for the presence of a specific feature in a time series, such as nonlinearity or a causal linkage, by comparing the observed data to a population of artificially generated "surrogate" datasets. These surrogates are carefully constructed to preserve certain statistical properties of the original data (the constraints) while randomizing or destroying the specific feature of interest (the null hypothesis). If a discriminating statistic computed for the original data is significantly different from the distribution of that same statistic computed for the surrogate ensemble, one can reject the null hypothesis and conclude that the feature is indeed present in the data.



### 2.2 The Centrality of the Null Hypothesis



The most critical aspect of the surrogate data method is that each surrogate generation algorithm is explicitly designed to test a specific, well-defined null hypothesis (H0). The choice of algorithm is therefore not a generic technical step but a precise formulation of the scientific question being asked. Common null hypotheses tested in the literature include:

1. H0: The data consists of uncorrelated, independent and identically distributed (i.i.d.) noise.
2. H0: The data is a realization of a stationary, linear, Gaussian stochastic process.
3. H0: The data is a static, monotonic nonlinear transformation of a linearly filtered noise process.
4. H0: The data originates from a periodic orbit with superimposed uncorrelated noise.



### 2.3 The General Procedure



The workflow for applying surrogate data testing is universal and follows a structured, step-by-step process :

1. **Formulate H0**: A clear and specific null hypothesis is stated. For CCM, this is often "the observed cross-map skill arises from non-causal statistical properties (e.g., shared linear correlation or seasonality) rather than a true causal link."
2. **Choose Surrogate Algorithm**: An algorithm is selected that generates surrogate datasets consistent with the formulated H0.
3. **Generate Ensemble**: A large ensemble of surrogate datasets (typically hundreds or thousands) is generated to create a robust null distribution.
4. **Compute Discriminating Statistic**: A chosen statistic, which for CCM is the cross-map skill ρ, is calculated for the original data and for each of the surrogate datasets.
5. **Compare and Infer**: The statistic from the original data is compared to the distribution of statistics from the surrogate ensemble. If the original value is an extreme outlier (e.g., falls above the 95th percentile of the null distribution), H0 is rejected with a corresponding level of statistical significance (a p-value).



## 3. A Taxonomy of Surrogate Generation Methods Relevant to CCM



The scientific literature describes a variety of surrogate generation algorithms, each testing a different null hypothesis. The selection of an appropriate algorithm is a critical decision that defines the precise nature of the causal inquiry.



### 3.1 Random Permutation (Shuffle) Surrogates



- **Algorithm**: This is the simplest method, involving a random shuffling of the time points of the original data series.
- **Properties**: This procedure perfectly preserves the amplitude distribution of the data (i.e., the mean, variance, and all higher-order moments are identical to the original). However, it completely destroys any temporal structure, including autocorrelation and cyclical patterns.
- **Null Hypothesis (H0)**: The data is independent and identically distributed (i.i.d.) noise drawn from an arbitrary distribution.
- **Caveat for CCM**: This method is strongly cautioned against for analyzing most real-world time series, particularly in fields like ecology. The underlying assumption of temporal independence is rarely met. Studies have shown that using this test for causal inference can lead to extremely high false-positive rates (30-92%), as any genuine temporal structure in the data will be incorrectly identified as a significant deviation from the i.i.d. null hypothesis. It is often a misleading default option in software packages.



### 3.2 Linear Process Surrogates: Phase Randomized (Fourier Transform) Surrogates



- **Algorithm**: This method operates in the frequency domain. It involves: (1) computing the Fast Fourier Transform (FFT) of the time series; (2) randomizing the phase components of the resulting complex numbers while keeping their amplitudes unchanged; and (3) computing the inverse FFT to return to the time domain.
- **Properties**: By preserving the Fourier amplitudes, this method precisely preserves the power spectrum of the original series. The power spectrum is equivalent to the autocorrelation function, meaning these surrogates have the same linear correlation structure as the original data. The randomization of phases destroys any nonlinear information or higher-order correlations.
- **Null Hypothesis (H0)**: The data is a realization of a stationary, linear, Gaussian stochastic process.
- **Caveat for CCM**: This method is effective when the data reasonably conforms to the assumptions of being Gaussian and stationary. However, its performance can degrade for strongly non-Gaussian or non-stationary data, which are common in many biological and economic systems, potentially leading to incorrect inferences.



### 3.3 Surrogates for Non-Gaussian Processes: AAFT and IAAFT



- **Amplitude Adjusted Fourier Transform (AAFT)**: This method is a refinement of phase randomization designed for non-Gaussian time series. The algorithm first rescales the original data to have a Gaussian distribution, then performs phase randomization on this Gaussianized series, and finally rescales the result back to match the original data's amplitude distribution. It aims to preserve both the power spectrum and the amplitude distribution, though the match to the power spectrum is often imperfect.
- **Iterated Amplitude Adjusted Fourier Transform (IAAFT)**: This is a more robust and widely preferred method that improves upon AAFT. IAAFT is an iterative algorithm that alternates between two steps until convergence: (1) imposing the correct power spectrum by substituting the Fourier amplitudes of the surrogate with those of the original data, and (2) imposing the correct amplitude distribution by rank-ordering the surrogate to match the original data's values. This iterative refinement produces surrogates that provide a much better simultaneous match to both the power spectrum and the amplitude distribution of the original series.
- **Null Hypothesis (H0)**: The data is a static, monotonic (and possibly nonlinear) transformation of a stationary, linear, Gaussian process. This is a more general and often more realistic null hypothesis for real-world data than that tested by simple phase randomization.



### 3.4 Dynamics-Preserving Surrogates: The Twin Surrogate Method



- **Algorithm**: This is a sophisticated method based on the concept of recurrences in the reconstructed state space. The algorithm identifies "twins"—pairs of points in the state-space trajectory that have identical or nearly identical neighborhoods. A new surrogate trajectory is then generated by starting at a random point and, whenever a twin is encountered, randomly choosing to follow the future of either the original point or its twin.
- **Properties**: Unlike other methods that preserve statistical properties, the Twin Surrogate method preserves the underlying dynamics of the system. It effectively creates an independent realization or a new trajectory on the same system attractor, starting from different initial conditions.
- **Null Hypothesis (H0)**: The two observed time series (X and Y) originate from two independent, uncoupled systems that happen to share the same underlying dynamical rules. This provides a powerful test for coupling between two complex systems.



### 3.5 Specialized Surrogates for Confounding Factors



- **Algorithm**: These methods are tailored to test for causality in the presence of a known, strong confounding variable, most commonly a seasonal or other cyclical driver. A typical procedure involves: (1) calculating the average cycle (e.g., the mean value for each month of the year); (2) subtracting this cycle from the original data to obtain a series of residuals; (3) randomizing these residuals (e.g., via shuffling or IAAFT); and (4) adding the average cycle back to the randomized residuals.
- **Properties**: This procedure generates surrogates that preserve the mean cyclical trend of the original data but destroy any finer-scale temporal information within that cycle, where the direct causal link might reside.
- **Null Hypothesis (H0)**: The observed cross-map skill between X and Y is entirely explained by their shared, synchronous response to a common cyclical driver, and there is no additional causal link between them.

The taxonomy of these methods reveals that there is no universally "best" surrogate algorithm. The choice is a deliberate scientific decision that precisely defines the question being asked. A researcher is not just vaguely "testing for causality," but rather testing against a specific, model-based null hypothesis. For instance, in an ecological study of algae blooms and water temperature, using Random Shuffle surrogates would test the trivial hypothesis: "Is this relationship more structured than random noise?" Using Phase Randomized surrogates tests a more interesting hypothesis: "Is there a *nonlinear* causal link, or can the relationship be explained by the linear properties of each series alone?" Using IAAFT surrogates refines this further: "Is there a *dynamic* causal link, or can it be explained by the linear properties and specific value distributions?" Finally, using Seasonal surrogates addresses the most practical question: "Is there a causal link *beyond* the fact that both algae and temperature follow the same annual cycle?". The selection of the surrogate method is therefore not a technical afterthought but a primary step in defining the scientific hypothesis.

| Method Name               | Algorithm Summary                                            | Preserved Properties                       | Randomized Properties                                       | Null Hypothesis (H0)                                         | Key Assumptions & Caveats for CCM                            |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Random Permutation**    | Randomly shuffles the temporal order of data points.         | Amplitude distribution.                    | Temporal order, autocorrelation.                            | The data is independent and identically distributed (i.i.d.) noise. | Assumes temporal independence; leads to extremely high false-positive rates for autocorrelated data. |
| **Phase Randomized (FT)** | Randomizes phases in the Fourier domain while keeping amplitudes constant. | Power spectrum (autocorrelation).          | Nonlinear phase information.                                | The data is a stationary, linear, Gaussian process.          | Assumes Gaussianity and stationarity; performance degrades for non-Gaussian/non-stationary data. |
| **IAAFT**                 | Iteratively adjusts the surrogate to match both the power spectrum and amplitude distribution of the original data. | Power spectrum and amplitude distribution. | Nonlinear phase information.                                | The data is a static, nonlinear transformation of a stationary, linear, Gaussian process. | A more robust and general null hypothesis for non-Gaussian data, but still assumes stationarity. |
| **Twin Surrogates**       | Identifies "twins" in state space and generates new trajectories by jumping between their futures. | System dynamics, attractor geometry.       | The specific trajectory (causal linkage to another system). | The observed series are from two uncoupled systems with identical dynamics. | Powerful test for coupling, but computationally intensive and requires sufficient data to find twins. |
| **Seasonal Surrogates**   | Randomizes residuals after removing the average seasonal cycle, then adds the cycle back. | Mean seasonal cycle.                       | Sub-seasonal causal links and temporal order within cycles. | The observed relationship is due solely to a shared seasonal driver. | Specifically designed to test for confounding by seasonality; not a general-purpose method. |



## 4. Integration and Application: Using Surrogates to Validate CCM Results





### 4.1 The Primary Rationale: Assessing the Statistical Significance of Cross-Map Skill (ρ)



The most fundamental and common application of surrogate data with CCM is to assess the statistical significance of the observed cross-map skill, ρ. A technique to assess the significance of a CCM result is to compare the calculated ρ to the distribution of ρ values computed from an ensemble of random realizations (surrogates) of the time series. A causal link is considered statistically significant if the observed 

ρ value is an extreme outlier relative to this null distribution, typically exceeding the 95th percentile. This procedure provides a formal p-value for the causal inference, transforming the CCM output from a descriptive statistic into a rigorous hypothesis test.



### 4.2 A Step-by-Step Guide to Significance Testing with CCM and Surrogates



The practical implementation of significance testing with CCM follows a clear workflow:

1. Perform CCM on the original pair of time series (X, Y) to determine the observed cross-map skill, ρobs, for the direction of interest (e.g., X→Y, which involves predicting Y from Mx).
2. Select a surrogate generation algorithm that is appropriate for the data and the specific null hypothesis being tested (e.g., IAAFT for non-Gaussian, autocorrelated data).
3. Generate a large ensemble of N surrogate time series for the putative causal variable (e.g., Xsurr,1,…,Xsurr,N).
4. For each surrogate Xsurr,i, perform CCM to predict the original response variable Y. This generates a null distribution of cross-map skills, ρnull,1,…,ρnull,N.
5. The statistical significance (p-value) is calculated by determining the proportion of the null distribution that is as extreme or more extreme than the observed value. A common formula is p=(r+1)/(N+1), where r is the number of surrogate skills (ρnull,i) that are greater than or equal to ρobs.
6. If this p-value is below a pre-defined significance level (e.g., α=0.05), the null hypothesis is rejected, and the conclusion is drawn that the observed cross-map skill is statistically significant.



### 4.3 Combating Spurious Causality



A more advanced and critical application of surrogate testing is to diagnose and mitigate spurious causality. Strong coupling, dynamic resonance, or a powerful common external driver (like seasonality) can lead to a phenomenon known as generalized synchrony between two variables. This synchrony can cause the CCM algorithm to produce high, convergent cross-map skill in both directions, creating a false signal of bidirectional causality.

Specialized surrogate methods are the primary tool for addressing this problem. For example, in the presence of strong seasonality, seasonal surrogates can be generated that share the exact same seasonal signal as the original data but are otherwise randomized. By comparing the original CCM skill to the null distribution from these seasonal surrogates, a researcher can test whether the observed causal link is significantly stronger than what would be expected from the shared seasonal driver alone. If the observed 

ρ falls within the distribution of the seasonal surrogates, the apparent causal link is dismissed as a spurious artifact of the common driver.



### 4.4 Case Studies from the Literature



The integrated framework of CCM and surrogate data testing has been successfully applied in numerous research contexts:

- **Ecology and Climatology**: In studies examining the influence of sea surface temperatures (SSTs) on snow water equivalent (SWE), seasonal surrogates are used to establish that the detected causal link is not merely an artifact of both variables following annual cycles. The CCM skill must significantly exceed the skill obtained from surrogates that retain the seasonal trend but erase sub-seasonal causal links.
- **Neuroscience**: When investigating dynamical coupling between brain signals, such as EEG and fNIRS, surrogate data analysis is crucial. By generating acausal pairs of time series through random permutation and calculating the CCM statistic on them, researchers create a null distribution. This allows them to reject the null hypothesis that an observed directional influence occurred by chance and provides a robust statistic for the coupling directionality.
- **Neural Spike Trains**: For point-process data like neural spike trains, where standard CCM is not directly applicable, modified CCM frameworks have been developed. In these cases, the powerful Twin Surrogate method is used to generate surrogate spike trains that preserve the underlying neuronal dynamics. The statistical significance of the cross-prediction accuracy is then assessed against these dynamically realistic surrogates to test for causal connections between neurons.



## 5. Methodological Considerations and Best Practices





### 5.1 Choosing the Right Surrogate: Matching the Null Hypothesis to the Scientific Question



The primary best practice is to recognize that the choice of surrogate method is a central part of the scientific inquiry. There is no universally superior algorithm; the method must be deliberately chosen to construct a null hypothesis that is relevant to the research question and appropriate for the data's characteristics. This choice and its justification should be explicitly stated in any study employing CCM.



### 5.2 The Perils of Inappropriate Surrogates



A mismatch between the surrogate method and the data can lead to erroneous conclusions. Using an overly simplistic method, such as random shuffling for data that is clearly autocorrelated, will almost always lead to a rejection of the null hypothesis. This results in a high rate of false positives, where any temporal structure is incorrectly flagged as significant causality. Conversely, using an overly constrained surrogate when a simpler null model is more appropriate might fail to detect a true causal signal, leading to false negatives.



### 5.3 The Impact of Noise, Non-Stationarity, and Coupling Strength



The validity of CCM and surrogate testing is subject to several important caveats:

- **Noise**: The performance of CCM is known to be sensitive to both process noise (stochasticity in the system's dynamics) and measurement noise. High levels of noise can obscure the underlying attractor and disrupt the detection of true causal links, making it difficult to distinguish a discontinuous map (no causality) from a noisy but continuous one (causality).
- **Non-Stationarity**: Most widely used surrogate algorithms (FT, AAFT, IAAFT) are designed for stationary time series. If the original data is non-stationary, a rejection of the null hypothesis could be due to this mismatch in stationarity rather than the presence of a true causal link. This is a significant challenge, as CCM itself can be confounded by certain types of non-stationarity, such as non-reverting dynamics where the system does not revisit past states.
- **Coupling Strength**: CCM performs best for systems with weak to moderate coupling. In cases of very strong coupling, the dynamics of the two variables can become so synchronized that their reconstructed manifolds become nearly identical. This can make it difficult to distinguish the direction of causality and may lead to spurious bidirectional results.

A subtle but critical aspect of the testing procedure concerns which variable in the causal pair should be used as the template for surrogate generation. When testing the significance of a causal link from X→Y, the choice of whether to generate surrogates of X (the presumed causer) or Y (the presumed causee) can dramatically alter the test's validity. If X truly drives Y, the dynamics of Y are a composite of its own internal processes plus the imprinted influence of X. Generating surrogates of Y using a dynamics-preserving method (like Twin Surrogates) may inadvertently preserve this "driven" characteristic, contaminating the null model and leading to a high false-positive rate. In contrast, generating surrogates of the independent driver, X, creates a set of time series that are dynamically similar to the original driver but have no causal connection to Y. Testing the cross-map skill from these X-surrogates to the original Y provides a much cleaner and more accurate test of the null hypothesis. This leads to a crucial best practice: when testing the significance of the causal link X→Y, surrogates should be generated from the presumed causal variable, X.



## 6. Conclusion and Future Directions



This review has established that surrogate data testing is not an optional or secondary validation step but an essential and integral component of the Convergent Cross Mapping workflow. It provides the necessary statistical framework to move beyond a descriptive measure of cross-map skill to a formal, rigorous test of causal inference. By allowing researchers to assign statistical significance to their findings and, critically, to diagnose and mitigate spurious causality arising from confounding factors like synchrony and common drivers, surrogate testing ensures the robustness and credibility of CCM-based causal claims.

Based on the synthesis of the scientific literature, a set of best practices for researchers emerges:

- Always employ surrogate data testing to evaluate the statistical significance of CCM results.
- Explicitly state the chosen surrogate method and provide a clear justification for why its corresponding null hypothesis is appropriate for the scientific question and the data at hand.
- Avoid simplistic methods like random shuffling for time series with known temporal dependencies, as this leads to a high likelihood of false-positive results.
- Utilize specialized surrogates (e.g., seasonal surrogates) to directly test for and rule out the influence of known, strong confounding variables.
- When testing a directional causal link X→Y, always generate the surrogate ensemble from the presumed causal variable, X, to ensure a valid null model.
- Remain cognizant of the inherent limitations of the methodology, particularly its sensitivity to high levels of noise, non-stationarity, and very strong coupling.

The field continues to evolve, with promising future directions aimed at addressing current limitations. This includes the development of more sophisticated surrogate methods capable of handling non-stationary data , the integration of CCM and surrogate testing into more comprehensive frameworks like Bayesian modeling (e.g., GP-CCM) to yield more robust statistics , and the ongoing need for systematic, comparative studies of different causality detection methods on benchmark datasets to better delineate their respective strengths, weaknesses, and domains of applicability.

------



# 代理数据检验在收敛交叉映射中的应用：一种用于稳健因果推断的方法学综述



**摘要**

本报告全面综述了代理数据检验与收敛交叉映射（CCM）相结合在非线性动力系统因果推断中的应用。我们首先阐明了植根于塔肯斯定理的CCM的理论基础，以及作为一种非参数假设检验方法的代理数据的基本原理。接着，我们详细分类了代理数据生成算法——从简单的随机置换到如迭代振幅调整傅里叶变换（IAAFT）和孪生代理等复杂方法——并分析了每种方法旨在检验的具体零假设。报告的核心部分聚焦于这些代理方法的实际应用，以评估CCM交叉映射技巧的统计显著性，并至关重要地，检测和减轻由同步、共同驱动因素及其他混淆因素引起的伪因果关系。通过综合科学文献，我们确立了最佳实践，指出了常见陷阱，并讨论了选择合适代理方法的深远影响。我们的结论是，代理数据检验并非一个可有可无的附加步骤，而是使用CCM实现稳健和可信因果推断不可或缺的组成部分。



## 引言



在复杂系统中，从观测时间序列中区分真实因果关系与纯粹相关性是一项根本性的科学挑战。从生态学、气候学到神经科学和金融学等领域，都充满了以非线性、不可分和协同动力学为特征的系统，传统的线性因果推断方法在这些系统中往往力不从心 。收敛交叉映射（CCM）作为一种源于动力系统理论的强大、无需方程的技术应运而生，专门用于在这些具有挑战性的情景中检测因果关系 。其核心前提是，一个因果变量会在受影响的变量上留下信息的“印记”，这可以通过对系统状态的熟练交叉预测来检测 。

然而，任何因果关系的主张，无论检测方法的复杂程度如何，都需要严格的统计验证。CCM算法的输出是一个相关系数 ρ，它量化了交叉预测的技巧 。一个高的 

ρ 值暗示了因果联系，但并非确凿证据。这就引出了一个关键问题：这个值必须多高才能被认为是统计显著的？没有一个正式的统计框架，任何为 ρ 选定的阈值都将是任意的。此外，诸如强耦合、动态共振或共同外部驱动因素（例如，季节性）的影响等现象，都可能在变量之间引起同步，从而产生高的 ρ 值，造成直接因果关系的假象，这通常被称为“伪双向因果关系” 。

这种对统计严谨性的迫切需求引入了代理数据方法，这是一种基于蒙特卡洛的非参数假设检验方法 。通过生成一个遵循特定零假设（例如，“不存在因果关系”）的人工时间序列集合，该方法提供了一个零分布，可以用来与观测结果进行比较 。这种比较允许计算p值，从而正式评估观测到的交叉映射技巧是否是一个统计上显著的发现，或者仅仅是数据固有属性或混淆因素的产物。因此，使用代理数据将CCM分析从一种描述性度量提升为一种正式的统计检验，使其成为该方法论不可或备的组成部分。

本报告基于对科学文献的全面回顾，详尽地综合了如何、为何以及哪些代理数据方法与CCM结合使用。报告将首先详细介绍CCM的理论基础，然后对相关的代理生成技术进行分类，最后详细阐述它们在稳健因果推断中的综合应用。



## 1. 收敛交叉映射的理论基础





### 1.1 状态空间重构与塔肯斯定理



CCM的理论基石是从单个观测时间序列重构动力系统多维状态空间的能力。一个系统随时间的行为可以被看作是在一个称为流形的几何对象上的轨迹，流形上的每一点代表系统的一个唯一状态 。塔肯斯嵌入定理提供了数学依据，证明在一般条件下，可以从该系统的单个时间序列观测中重构出一个拓扑等价的“影子流形” 。

这种重构是通过时间延迟嵌入实现的。对于一个时间序列 X，在每个时间点 Xt 创建一个 E 维状态向量（或延迟向量），方法是将其当前值与 E−1 个过去的值相结合，每个值之间相隔一个时间延迟 τ。得到的向量是 x~t=(Xt,Xt−τ,Xt−2τ,…,Xt−(E−1)τ) 。所有这些向量的集合构成了影子流形 

Mx。嵌入维度 E 和时间延迟 τ 是关键参数，存在确定其最优值的方法，以确保对系统动力学的忠实重构 。



### 1.2 交叉映射的逻辑



CCM利用了塔肯斯定理的一个关键推论：如果两个变量 X 和 Y 是同一个耦合动力系统的一部分，它们各自重构的影子流形 Mx 和 My 将彼此微分同胚，并且与真实的系统流形微分同胚 。这种拓扑等价性意味着两个流形上的点之间存在一对一的映射。

CCM的核心前提是，如果变量 X 对变量 Y 有因果影响，那么关于 X 的历史信息必然被编码在 Y 的动力学中 。因此，从 

Y 重构的流形（My）上的邻近点将对应于从 X 重构的流形（Mx）上的邻近点。这一特性使得可以利用 My上的点库来对 X 的状态进行熟练的预测。这个过程被称为交叉映射，表示为 X^∣My 。这种交叉预测通常是不对称的。如果 

X 单向驱动 Y，那么 Y 将包含关于 X 的信息，使得预测 X^∣My 是熟练的。然而，X 将不包含关于 Y 的信息，因此反向预测 Y^∣Mx 将不熟练 。



### 1.3 收敛原则



将CCM与简单相关区分开来的决定性特征是收敛原则。交叉映射预测的技巧，由预测值（X^∣My）与实际值（X）之间的皮尔逊相关系数（ρ）量化，取决于用于构建流形的点的密度 。

CCM中因果关系的决定性检验是观察“收敛”：随着用于构建流形的库长度 L（时间点数量）的增加，重构的流形变得更密集、更完整。如果存在因果联系，这种增加的密度可以更准确地识别最近邻，因此交叉映射预测技巧 ρ 应该系统性地增加，并最终在一个高值处饱和 。这种收敛的特征被认为是因果关系的必要条件，因为它表明这种关系是共享系统动力学的基本属性，而非偶然的相关性 。



### 1.4 与格兰杰因果的区别及应用领域



CCM与格兰杰因果（GC）有根本不同。GC是一个基于可预测性和时间优先性的统计概念，适用于随机、可分的系统（其中因果影响是独立的），而CCM则植根于动力系统理论，专门为确定性、不可分的系统设计，其中变量可能具有协同效应 。CCM可以识别使用标准线性方法可能看起来不相关的变量之间的联系。

CCM的效用已在广泛的科学领域得到证明。应用包括识别生态系统中的因果关系（例如，捕食者-猎物动力学、沙丁鱼种群与海洋温度）、气候学（例如，宇宙射线与全球温度）、金融学（例如，股价与交易量）、神经科学（例如，大脑连接性）和水文学（例如，地下水与河流流量）。



## 2. 用于假设检验的代理数据方法





### 2.1 方法哲学：统计反证法



代理数据检验是一种非参数的、基于蒙特卡洛的技术，其功能类似于统计上的反证法 。其核心逻辑是通过将观测数据与一个人为生成的“代理”数据集群体进行比较，来检验时间序列中是否存在特定特征，如非线性或因果联系。这些代理数据经过精心构建，以保留原始数据的某些统计特性（约束条件），同时随机化或破坏感兴趣的特定特征（零假设）。如果为原始数据计算的判别统计量与为代理数据集集合计算的该统计量的分布有显著差异，就可以拒绝零假设，并得出结论认为该特征确实存在于数据中 。



### 2.2 零假设的核心地位



代理数据方法最关键的方面是，每种代理生成算法都是为检验一个特定的、明确定义的零假设（H0）而设计的 。因此，算法的选择不是一个通用的技术步骤，而是对所问科学问题的精确表述。文献中常见的检验零假设包括：

1. H0：数据由不相关的、独立同分布（i.i.d.）的噪声组成 。
2. H0：数据是平稳、线性、高斯随机过程的实现 。
3. H0：数据是线性滤波噪声过程的静态、单调非线性变换 。
4. H0：数据源于一个叠加了不相关噪声的周期轨道 。



### 2.3 一般流程



应用代理数据检验的工作流程是通用的，并遵循一个结构化的、分步的过程 ：

1. **陈述 H0**：明确且具体地陈述零假设。对于CCM，这通常是“观测到的交叉映射技巧源于非因果的统计特性（例如，共享的线性相关性或季节性），而非真实的因果联系。”
2. **选择代理算法**：选择一个能够生成与所陈述的 H0 一致的代理数据集的算法。
3. **生成集合**：生成大量的代理数据集（通常是数百或数千个），以创建一个稳健的零分布。
4. **计算判别统计量**：为原始数据和每个代理数据集计算一个选定的统计量，对于CCM而言，这个统计量就是交叉映射技巧 ρ。
5. **比较与推断**：将原始数据的统计量与代理数据集集合的统计量分布进行比较。如果原始值是一个极端异常值（例如，落在零分布的95百分位数以上），则以相应的统计显著性水平（p值）拒绝 H0 。



## 3. 与CCM相关的代理生成方法分类



科学文献描述了多种代理生成算法，每种算法检验不同的零假设。选择合适的算法是一个关键决策，它定义了因果探究的精确性质。



### 3.1 随机置换（洗牌）代理



- **算法**：这是最简单的方法，涉及对原始数据系列的时间点进行随机洗牌 。
- **特性**：此过程完美地保留了数据的振幅分布（即均值、方差和所有高阶矩与原始数据相同）。然而，它完全破坏了所有时间结构，包括自相关和周期性模式 。
- **零假设 (H0)**：数据是来自任意分布的独立同分布（i.i.d.）噪声 。
- **对CCM的警示**：强烈不建议使用此方法分析大多数现实世界的时间序列，尤其是在生态学等领域。时间独立性的基本假设很少得到满足。研究表明，使用此检验进行因果推断可能导致极高的假阳性率（30-92%），因为数据中任何真实的时间结构都会被错误地识别为与i.i.d.零假设的显著偏离。它常常是软件包中一个误导性的默认选项 。



### 3.2 线性过程代理：相位随机化（傅里叶变换）代理



- **算法**：此方法在频域中操作。它包括：（1）计算时间序列的快速傅里叶变换（FFT）；（2）随机化所得复数的相位分量，同时保持其振幅不变；（3）计算逆FFT以返回时域 。
- **特性**：通过保留傅里叶振幅，此方法精确地保留了原始序列的功率谱。功率谱等同于自相关函数，意味着这些代理数据具有与原始数据相同的线性相关结构。相位的随机化破坏了任何非线性信息或高阶相关性 。
- **零假设 (H0)**：数据是平稳、线性、高斯随机过程的实现 。
- **对CCM的警示**：当数据合理地符合高斯和平稳的假设时，此方法是有效的。然而，对于强非高斯或非平稳数据，其性能可能会下降，这在许多生物和经济系统中很常见，可能导致不正确的推断 。



### 3.3 非高斯过程的代理：AAFT和IAAFT



- **振幅调整傅里叶变换（AAFT）**：这是为非高斯时间序列设计的相位随机化的改进方法。该算法首先将原始数据重新缩放为高斯分布，然后对这个高斯化序列进行相位随机化，最后将结果重新缩放以匹配原始数据的振幅分布 。它旨在同时保留功率谱和振幅分布，尽管对功率谱的匹配通常不完美 。
- **迭代振幅调整傅里叶变换（IAAFT）**：这是一种更稳健且被广泛偏爱的方法，它改进了AAFT。IAAFT是一种迭代算法，交替执行两个步骤直到收敛：（1）通过将代理的傅里叶振幅替换为原始数据的振幅来施加正确的功率谱，以及（2）通过对代理进行排序以匹配原始数据的值来施加正确的振幅分布 。这种迭代优化产生的代理数据能更好地同时匹配原始序列的功率谱和振幅分布 。
- **零假设 (H0)**：数据是平稳、线性、高斯过程的静态、单调（可能非线性）变换 。对于现实世界的数据，这是一个比简单相位随机化所检验的更通用且通常更现实的零假设。



### 3.4 动力学保持代理：孪生代理方法



- **算法**：这是一种基于重构状态空间中复现概念的复杂方法。该算法识别“孪生点”——状态空间轨迹中具有相同或几乎相同邻域的点对。然后，通过从一个随机点开始，并在遇到孪生点时随机选择跟随原始点或其孪生点的未来轨迹，来生成一个新的代理轨迹 。
- **特性**：与其他保留统计特性的方法不同，孪生代理方法保留了系统的基本动力学。它有效地创建了同一动力系统的一个独立实现或一个新轨迹，从不同的初始条件开始 。
- **零假设 (H0)**：两个观测到的时间序列（X 和 Y）源于两个独立的、非耦合的系统，这两个系统恰好共享相同的基本动力学规则 。这为检验两个复杂系统之间的耦合提供了一个强有力的检验。



### 3.5 针对混淆因素的专门代理



- **算法**：这些方法是为在存在已知的、强烈的混淆变量（最常见的是季节性或其他周期性驱动因素）的情况下检验因果关系而量身定制的。一个典型的程序包括：（1）计算平均周期（例如，一年中每个月的平均值）；（2）从原始数据中减去这个周期以获得残差序列；（3）随机化这些残差（例如，通过洗牌或IAAFT）；（4）将平均周期加回到随机化的残差上 。
- **特性**：此过程生成的代理数据保留了原始数据的平均周期性趋势，但破坏了该周期内任何更精细的时间信息，而直接的因果联系可能就存在于这些信息中 。
- **零假设 (H0)**：观测到的 X 和 Y 之间的交叉映射技巧完全由它们对共同周期性驱动因素的同步响应所解释，它们之间没有额外的因果联系 。

这些方法的分类揭示了不存在普遍“最佳”的代理算法。选择是一种深思熟虑的科学决策，它精确地定义了所要探究的问题。研究人员不仅仅是在模糊地“检验因果关系”，而是在针对一个具体的、基于模型的零假设进行检验。例如，在一个关于藻类爆发和水温的生态学研究中，使用随机洗牌代理将检验一个微不足道的假设：“这种关系是否比随机噪声更有结构？”使用相位随机化代理检验一个更有趣的假设：“是否存在*非线性*的因果联系，或者这种关系能否仅由每个序列的线性特性来解释？”使用IAAFT代理则进一步细化了这个问题：“是否存在*动态*的因果联系，或者它能否由线性特性和特定的值分布来解释？”最后，使用季节性代理解决了最实际的问题：“是否存在*超越*藻类和温度都遵循相同年度周期的因果联系？” 。因此，代理方法的选择不是一个技术性的事后考虑，而是定义科学假设的首要步骤。

| 方法名称            | 算法摘要                                                     | 保留的特性               | 随机化的特性                         | 检验的零假设 (H0)                              | 对CCM的关键假设与警示                                       |
| ------------------- | ------------------------------------------------------------ | ------------------------ | ------------------------------------ | ---------------------------------------------- | ----------------------------------------------------------- |
| **随机置换**        | 随机打乱数据点的时间顺序。                                   | 振幅分布。               | 时间顺序、自相关。                   | 数据是独立同分布（i.i.d.）的噪声。             | 假设时间独立性；对于自相关数据导致极高的假阳性率 。         |
| **相位随机化 (FT)** | 在傅里叶域中随机化相位，同时保持振幅不变。                   | 功率谱（自相关）。       | 非线性相位信息。                     | 数据是一个平稳、线性的高斯过程。               | 假设高斯性和平稳性；对于非高斯/非平稳数据性能下降 。        |
| **IAAFT**           | 迭代调整代理数据以同时匹配原始数据的功率谱和振幅分布。       | 功率谱和振幅分布。       | 非线性相位信息。                     | 数据是平稳、线性、高斯过程的静态、非线性变换。 | 对于非高斯数据是一个更稳健和通用的零假设，但仍假设平稳性 。 |
| **孪生代理**        | 在状态空间中识别“孪生点”，并通过在它们的未来轨迹之间跳转来生成新轨迹。 | 系统动力学、吸引子几何。 | 特定轨迹（与另一系统的因果联系）。   | 观测序列来自两个具有相同动力学的非耦合系统。   | 强有力的耦合检验，但计算密集且需要足够数据找到孪生点 。     |
| **季节性代理**      | 去除平均季节周期后随机化残差，然后加回周期。                 | 平均季节周期。           | 周期内的亚季节性因果联系和时间顺序。 | 观测到的关系仅由共享的季节性驱动因素引起。     | 专门设计用于检验季节性混淆；非通用方法 。                   |



## 4. 整合与应用：使用代理数据验证CCM结果





### 4.1 主要理由：评估交叉映射技巧 (ρ) 的统计显著性



代理数据与CCM最基本和常见的应用是评估观测到的交叉映射技巧 ρ 的统计显著性。评估CCM结果显著性的一种技术是将计算出的 ρ 与从时间序列的随机实现（代理）集合中计算出的 ρ 值分布进行比较 。如果观测到的 

ρ 值相对于这个零分布是一个极端异常值，通常超过95百分位数，则认为因果联系是统计显著的 。这个过程为因果推断提供了一个正式的p值，将CCM的输出从一个描述性统计量转变为一个严格的假设检验 。



### 4.2 CCM与代理数据显著性检验分步指南



CCM显著性检验的实际操作遵循一个清晰的工作流程：

1. 对原始时间序列对（X, Y）执行CCM，以确定感兴趣方向的观测交叉映射技巧，ρobs（例如，X→Y，这涉及从 Mx 预测 Y）。
2. 根据要检验的特定零假设和数据特性，选择一个合适的代理生成算法（例如，对于非高斯、自相关数据选择IAAFT）。
3. 为假定的因果变量生成一个包含 N 个代理时间序列的集合（例如，Xsurr,1,…,Xsurr,N）。
4. 对于每个代理 Xsurr,i，执行CCM以预测原始响应变量 Y。这将生成一个交叉映射技巧的零分布，ρnull,1,…,ρnull,N。
5. 通过确定零分布中与观测值一样极端或更极端的比例来计算统计显著性（p值）。一个常用的公式是 p=(r+1)/(N+1)，其中 r 是大于或等于 ρobs 的代理技巧（ρnull,i）的数量 。
6. 如果这个p值低于预先定义的显著性水平（例如，α=0.05），则拒绝零假设，并得出结论认为观测到的交叉映射技巧是统计显著的。



### 4.3 对抗伪因果关系



代理检验的一个更高级且至关重要的应用是诊断和减轻伪因果关系。强耦合、动态共振或强大的共同外部驱动因素（如季节性）可能导致两个变量之间出现一种称为广义同步的现象。这种同步可能导致CCM算法在两个方向上都产生高的、收敛的交叉映射技巧，从而造成双向因果关系的假象 。

专门的代理方法是解决这个问题的主要工具。例如，在存在强季节性的情况下，可以生成与原始数据共享完全相同季节信号但其他方面随机化的季节性代理。通过将原始CCM技巧与这些季节性代理产生的零分布进行比较，研究人员可以检验观测到的因果联系是否显著强于仅由共享季节性驱动因素所能解释的水平 。如果观测到的 

ρ 落在季节性代理的分布范围内，那么 apparent 的因果联系将被视为共同驱动因素的伪影而被排除。



### 4.4 文献案例研究



CCM与代理数据检验的综合框架已在众多研究背景中成功应用：

- **生态学与气候学**：在研究海面温度（SSTs）对雪水当量（SWE）影响的研究中，使用季节性代理来确定检测到的因果联系不仅仅是两个变量都遵循年度周期的产物。CCM技巧必须显著超过从保留季节趋势但消除亚季节性因果联系的代理中获得的技巧 。
- **神经科学**：在研究大脑信号（如EEG和fNIRS）之间的动态耦合时，代理数据分析至关重要。通过随机置换生成非因果的时间序列对，并对其计算CCM统计量，研究人员创建了一个零分布。这使他们能够拒绝观测到的方向性影响是偶然发生的零假设，并为耦合方向性提供一个稳健的统计量 。
- **神经脉冲序列**：对于像神经脉冲序列这样的点过程数据，标准CCM不直接适用，因此开发了改进的CCM框架。在这些情况下，使用强大的孪生代理方法生成保留基本神经元动力学的代理脉冲序列。然后，根据这些动态真实的代理评估交叉预测准确性的统计显著性，以检验神经元之间的因果连接 。



## 5. 方法论考量与最佳实践





### 5.1 选择正确的代理：将零假设与科学问题匹配



首要的最佳实践是认识到代理方法的选择是科学探究的核心部分。没有普遍优越的算法；必须审慎选择方法，以构建一个与研究问题相关且适合数据特性的零假设。在任何使用CCM的研究中，都应明确陈述这一选择及其理由。



### 5.2 不当代理的危害



代理方法与数据之间的不匹配可能导致错误的结论。对于明显自相关的数据使用过于简单的方法，如随机洗牌，几乎总会导致拒绝零假设。这会导致高假阳性率，任何时间结构都会被错误地标记为显著的因果关系 。相反，当一个更简单的零模型更合适时，使用一个高度约束的代理（如孪生代理）可能会未能检测到真实的因果信号，导致假阴性。



### 5.3 噪声、非平稳性和耦合强度的影响



CCM和代理检验的有效性受到几个重要注意事项的制约：

- **噪声**：已知CCM的性能对过程噪声（系统动力学中的随机性）和测量噪声都很敏感。高水平的噪声会掩盖底层的吸引子并干扰真实因果联系的检测，使得难以区分不连续的映射（无因果关系）和嘈杂但连续的映射（有因果关系）。
- **非平稳性**：大多数广泛使用的代理算法（FT、AAFT、IAAFT）都是为平稳时间序列设计的。如果原始数据是非平稳的，拒绝零假设可能是由于这种平稳性的不匹配，而不是真实的因果联系 。这是一个重大的挑战，因为CCM本身也可能被某些类型的非平稳性所混淆，例如系统不重访过去状态的非回复动力学 。
- **耦合强度**：CCM在弱到中等耦合的系统中表现最佳。在非常强耦合的情况下，两个变量的动力学可能变得如此同步，以至于它们重构的流形几乎相同。这可能使得难以区分因果关系的方向，并可能导致伪双向结果 。

检验过程的一个微妙但关键的方面涉及在因果对中应使用哪个变量作为代理生成的模板。在检验从 X→Y 的因果联系的显著性时，选择生成 X（假定的原因）或 Y（假定的结果）的代理，可以极大地改变检验的有效性。如果 X 确实驱动 Y，那么 Y 的动力学是其自身内部过程加上 X 印记影响的复合体。使用保留动力学的方法（如孪生代理）生成 Y 的代理，可能会无意中保留这种“被驱动”的特性，从而污染零模型并导致高假阳性率。相比之下，生成独立驱动因素 X 的代理，会创建一组与原始驱动因素动态相似但与 Y 没有因果联系的时间序列。检验从这些 X-代理到原始 Y 的交叉映射技巧，为零假设提供了一个更清晰、更准确的检验。这导致了一个至关重要的最佳实践：在检验因果联系 X→Y 的显著性时，应从假定的因果变量 X 生成代理 。



## 6. 结论与未来方向



本综述确立了代理数据检验并非一个可有可无或次要的验证步骤，而是收敛交叉映射工作流程中一个必不可少且不可或缺的组成部分。它提供了必要的统计框架，使我们能够从交叉映射技巧的描述性度量转向对因果推断的正式、严格的检验。通过让研究人员能够为其发现赋予统计显著性，并且至关重要地，诊断和减轻由同步和共同驱动因素等混淆因素引起的伪因果关系，代理检验确保了基于CCM的因果主张的稳健性和可信度。

基于对科学文献的综合，为研究人员提出了一套最佳实践：

- 始终使用代理数据检验来评估CCM结果的统计显著性。
- 明确陈述所选的代理方法，并为其相应的零假设为何适合科学问题和手头数据提供清晰的理由。
- 对于具有已知时间依赖性的时间序列，避免使用随机洗牌等简单化方法，因为这很可能导致假阳性结果。
- 利用专门的代理（例如，季节性代理）来直接检验和排除已知的、强烈的混淆变量的影响。
- 在检验方向性因果联系 X→Y 时，始终从假定的因果变量 X 生成代理集合，以确保一个有效的零模型。
- 注意该方法的内在局限性，特别是其对高水平噪声、非平稳性和非常强耦合的敏感性。

该领域在不断发展，旨在解决当前局限性的未来方向充满希望。这包括开发能够处理非平稳数据的更复杂的代理方法 ，将CCM和代理检验整合到更全面的框架中，如贝叶斯建模（例如，GP-CCM），以产生更稳健的统计数据 ，以及持续需要对不同因果检测方法（包括CCM与各种代理检验）在基准数据集上进行系统的比较研究，以更好地界定它们各自的优势、劣势和适用领域 。