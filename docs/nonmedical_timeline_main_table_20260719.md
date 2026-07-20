# Complete non-medical long-timeline comparison

All accuracy and recall values below are percentages. DialSim latency is the sample-weighted end-to-end QA latency in milliseconds. The average accuracy is the arithmetic mean over the four datasets. Average Evidence R@5 is the arithmetic mean over LoCoMo, LongMemEval, and RHELM and is shown only for methods that return ranked evidence.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">LoCoMo</th>
      <th colspan="2">LongMemEval</th>
      <th colspan="2">DialSim</th>
      <th colspan="2">RHELM</th>
      <th colspan="2">Average</th>
    </tr>
    <tr>
      <th>Answer Acc. ↑</th><th>Evidence R@5 ↑</th>
      <th>Answer Acc. ↑</th><th>Evidence R@5 ↑</th>
      <th>Answer Acc. ↑</th><th>Latency (ms) ↓</th>
      <th>Answer Acc. ↑</th><th>Evidence R@5 ↑</th>
      <th>Answer Acc. ↑</th><th>Evidence R@5 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Direct</td><td>25.50</td><td>--</td><td>14.20</td><td>--</td><td>28.90</td><td><strong>556.59</strong></td><td>14.10</td><td>--</td><td>20.67</td><td>--</td></tr>
    <tr><td>Static RAG</td><td>46.20</td><td>30.88</td><td><u>48.60</u></td><td>70.17</td><td>59.60</td><td>4323.89</td><td><u>30.42</u></td><td>16.55</td><td><u>46.21</u></td><td>39.20</td></tr>
    <tr><td>A-MEM</td><td>27.40</td><td>16.93</td><td>23.00</td><td>47.07</td><td>32.30</td><td><u>1402.30</u></td><td>15.86</td><td>7.53</td><td>24.64</td><td>23.84</td></tr>
    <tr><td>DDO</td><td>35.70</td><td>17.89</td><td>38.20</td><td>58.68</td><td><strong>60.60</strong></td><td>4228.69</td><td>26.67</td><td>17.03</td><td>40.29</td><td>31.20</td></tr>
    <tr><td>G-Memory</td><td>36.20</td><td>17.86</td><td>38.00</td><td>58.68</td><td>60.40</td><td>4982.01</td><td>27.20</td><td>17.03</td><td>40.45</td><td>31.19</td></tr>
    <tr><td>MemInsight</td><td>36.30</td><td>17.86</td><td>38.20</td><td>58.68</td><td><u>60.50</u></td><td>4997.45</td><td>27.36</td><td>17.03</td><td>40.59</td><td>31.19</td></tr>
    <tr><td>MemoryOS</td><td>35.90</td><td>17.86</td><td>38.20</td><td>58.68</td><td>60.40</td><td>5001.83</td><td>27.05</td><td>17.03</td><td>40.39</td><td>31.19</td></tr>
    <tr><td>Mem0</td><td><strong>61.60</strong></td><td><strong>87.62</strong></td><td><strong>49.40</strong></td><td><strong>89.74</strong></td><td>60.00</td><td>1427.19</td><td><strong>33.03</strong></td><td><strong>30.18</strong></td><td><strong>51.01</strong></td><td><strong>69.18</strong></td></tr>
    <tr><td>Letta/MemGPT</td><td>10.90</td><td>--</td><td>5.60</td><td>--</td><td>26.80</td><td>5625.03</td><td>12.11</td><td>--</td><td>13.85</td><td>--</td></tr>
    <tr><td><strong>MediMem</strong></td><td><u>52.50</u></td><td><u>71.76</u></td><td>39.80</td><td><u>79.53</u></td><td>58.60</td><td>3987.62</td><td>30.34</td><td><u>28.44</u></td><td>45.31</td><td><u>59.91</u></td></tr>
  </tbody>
</table>

Best entries are bold and second-best entries are underlined; ties receive the same marking. Cost/latency does not participate in the average.

The MediMem row is the single complete hybrid-BGE formal run selected by the fallback rule (highest average answer accuracy, then average Evidence R@5, then lower token use). It is not assembled from per-dataset best values. The full comparison is retained as an appendix result because Mem0 leads both aggregate metrics.

DDO, G-Memory, and MemInsight use a unified protocol wrapper and are not claimed as official native four-dataset reproductions. Letta/MemGPT uses the query-independent `uniform_session_coverage_v1` timeline protocol adapter with a 6500-character core-memory limit; this qualification belongs outside the table. Direct and Letta/MemGPT do not return ranked retrieval evidence, so Evidence R@5 is `--`. DialSim's official QA indices cannot be mapped to runtime source records, so its second metric is latency rather than Evidence R@5.

## LaTeX

```latex
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\caption{Complete non-medical long-timeline results on four data sources. Answer Acc. and Evidence R@5 are reported in percentage points. The MediMem row is one complete hybrid-BGE run, not a per-source combination. Best results are bold and second-best results are underlined.}
\label{tab:nonmedical_timeline_main}
\begin{tabular}{lcccccccccc}
\toprule
& \multicolumn{2}{c}{LoCoMo} & \multicolumn{2}{c}{LongMemEval} & \multicolumn{2}{c}{DialSim} & \multicolumn{2}{c}{RHELM} & \multicolumn{2}{c}{Average} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Method & Acc.$\uparrow$ & R@5$\uparrow$ & Acc.$\uparrow$ & R@5$\uparrow$ & Acc.$\uparrow$ & Lat.(ms)$\downarrow$ & Acc.$\uparrow$ & R@5$\uparrow$ & Acc.$\uparrow$ & R@5$\uparrow$ \\
\midrule
Direct & 25.50 & -- & 14.20 & -- & 28.90 & \textbf{556.59} & 14.10 & -- & 20.67 & -- \\
Static RAG & 46.20 & 30.88 & \underline{48.60} & 70.17 & 59.60 & 4323.89 & \underline{30.42} & 16.55 & \underline{46.21} & 39.20 \\
A-MEM & 27.40 & 16.93 & 23.00 & 47.07 & 32.30 & \underline{1402.30} & 15.86 & 7.53 & 24.64 & 23.84 \\
DDO & 35.70 & 17.89 & 38.20 & 58.68 & \textbf{60.60} & 4228.69 & 26.67 & 17.03 & 40.29 & 31.20 \\
G-Memory & 36.20 & 17.86 & 38.00 & 58.68 & 60.40 & 4982.01 & 27.20 & 17.03 & 40.45 & 31.19 \\
MemInsight & 36.30 & 17.86 & 38.20 & 58.68 & \underline{60.50} & 4997.45 & 27.36 & 17.03 & 40.59 & 31.19 \\
MemoryOS & 35.90 & 17.86 & 38.20 & 58.68 & 60.40 & 5001.83 & 27.05 & 17.03 & 40.39 & 31.19 \\
Mem0 & \textbf{61.60} & \textbf{87.62} & \textbf{49.40} & \textbf{89.74} & 60.00 & 1427.19 & \textbf{33.03} & \textbf{30.18} & \textbf{51.01} & \textbf{69.18} \\
Letta/MemGPT & 10.90 & -- & 5.60 & -- & 26.80 & 5625.03 & 12.11 & -- & 13.85 & -- \\
\midrule
\textbf{MediMem} & \underline{52.50} & \underline{71.76} & 39.80 & \underline{79.53} & 58.60 & 3987.62 & 30.34 & \underline{28.44} & 45.31 & \underline{59.91} \\
\bottomrule
\end{tabular}
\end{table*}
```

The raw merged values are stored in `outputs/nonmedical_timeline_main_table_20260719.csv`. The Mem0 comparability audit and the retained MediMem run both completed all 3805 predictions with passed validation, zero fallbacks, and zero missing judge results.
