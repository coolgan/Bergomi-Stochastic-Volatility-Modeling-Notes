# Bergomi 随机波动率建模中文精读笔记

本项目为Lorenzo Bergomi《Stochastic Volatility Modeling》的中文学习笔记站点。笔记按原书章节重写推导路线，补上公式之间的过渡、模型假设的含义，以及一些交易台视角下更容易忽略的风险管理问题。

在线阅读：

https://bergomi-stochastic-volatility-modeling-notes.pages.dev/

## 项目简介

Bergomi这本书着重讨论了以下问题：为什么 Black-Scholes 可以作为 P&L 会计工具使用？局部波动率为什么能精确校准却给出不自然的动态？前向方差模型如何把 vol of vol、偏斜和期限结构拆开处理？混合模型又在什么条件下才不会产生 P&L 泄漏？

这些笔记围绕这些问题展开。每章保留原书的数学结构，同时尽量把以下几件事说清楚：

- 公式从哪里来，哪些近似在起作用；
- 模型参数对应什么市场量或风险敞口；
- 静态微笑、动态偏斜、vol of vol、SSR 之间怎样互相约束；
- 哪些结论适合用于定价，哪些更适合作为风险管理和模型诊断工具。

当前站点共生成 12 篇章节笔记。

## 章节目录

- 第一章：导论（Introduction）
- 第二章：局部波动率模型（Local Volatility）
- 第三章：远期起始期权（Forward Start Options）
- 第四章：随机波动率导论（Stochastic Volatility Introduction）
- 第五章：方差互换（Variance Swaps）
- 第六章：单因子动态的典型案例——Heston 模型（Heston Model）
- 第七章：前向方差模型（Forward Variance Models）
- 第八章：随机波动率模型的微笑（Smile Of SV Models）
- 第九章：随机波动率模型的静态与动态性质的关联（Static Dynamic Properties）
- 第十章：股权微笑的成因（What Causes Equity Smiles）
- 第十一章：多资产随机波动率模型（Multi Asset SV）
- 第十二章：局部随机波动率模型（Local Stochastic Volatility）

## 阅读建议

如果是第一次系统读随机波动率模型，建议先读第 1、2 章，弄清楚 P&L 会计、局部波动率和市场模型的语言。第 4、6、7 章再进入随机波动率和前向方差模型。第 8、9 章适合连在一起读：前者讲微笑的扰动展开，后者讲静态偏斜和动态 SSR 的关系。第 12 章最好放到最后，它需要同时用到局部波动率和前向方差模型。

如果目标是交易或风控应用，可以优先看第 5、7、9、10、12 章。方差互换、前向方差、股权微笑成因和局部随机波动率模型，和实际期权账簿的风险解释更接近。

## 站点结构

本仓库保存的是已经构建好的静态网页，仓库根目录就是站点根目录。主要文件包括：

- `index.html`：章节入口页；
- `chapter-*.html`：各章笔记页面；
- `assets/`：CSS、页面脚本、KaTeX 样式和字体；
- `images/`：笔记中引用的图表。

网页由 Markdown 笔记构建生成，数学公式在构建阶段用 KaTeX 渲染，图片和字体都随仓库一起发布，不依赖外部 CDN。

## 说明

本项目是个人学习整理，不是原书的官方译本。笔记中的解释、补充例子和中文表述都可能带有整理者的理解偏差。涉及定价、对冲或风险管理的内容只用于学习讨论，不构成投资建议或交易建议。
