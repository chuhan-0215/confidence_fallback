"""术语注释：故事内 [[id|显示名]] 链接到本页条目。"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_HTML = ROOT / "glossary.html"
GLOSSARY_URL = "/others/000002/glossary.html"

TERM_PATTERN = re.compile(r"\[\[([a-z0-9-]+)\|([^\]]+)\]\]")

TERMS: dict[str, dict] = {
    "superposition": {
        "title": "叠加态（Superposition）",
        "one_liner": (
            "很多条搜索路径被压进同一个向量里同时维护。"
        ),
        "analogy": (
            "像脑子里同时记着「可能在 1 楼、2 楼、3 楼」而不是一次只想一层。"
        ),
        "body": (
            "来自论文 Reasoning by Superposition 的核心概念。"
            "模型在潜空间里同时维护多个搜索前沿的「叠加」表示，"
            "而不是一步一步写出离散推理链。可以把它想成："
            "很多条可能的图搜索路径被压缩进同一个向量里。"
        ),
    },
    "parallel-bfs": {
        "title": "并行 BFS",
        "one_liner": (
            "一次性维护多层搜索前沿，不是 A→B→C 串行。"
        ),
        "analogy": (
            "多队侦察兵同时占 1、2、3 层，而不是排队上楼。"
        ),
        "body": (
            "不是串行地「想完 A 再想 B」，而是在潜空间里一次性维护多层搜索前沿。"
            "实验曲线里 2→3 步的猛涨，常被解释为搜索前沿在这一步大幅展开。"
        ),
    },
    "latent-steps": {
        "title": "连续思维步数",
        "one_liner": (
            "思考标记的个数，本实验从 1 扫到 10。"
        ),
        "analogy": (
            "搜救队允许往上搜几层——层数越多越贵。"
        ),
        "body": (
            "prompt 中 latent token 的个数，也是本实验扫描的自变量（1–10 步）。"
            "步数越多，计算越贵，但准确率不一定更高。"
        ),
    },
    "coconut": {
        "title": "Coconut",
        "one_liner": (
            "用连续思维做图推理的 NeurIPS 模型。"
        ),
        "analogy": (
            "本实验用的那位「会在脑子里搜图」的选手。"
        ),
        "body": (
            "NeurIPS 2025 论文提出的模型，用连续思维做图上的可达性推理。"
            "本实验使用 HuggingFace 权重 checkpoint_300，在 ProsQA 上微调过。"
        ),
    },
    "checkpoint": {
        "title": "checkpoint_300",
        "one_liner": (
            "固定不动的模型权重，实验只改思考步数。"
        ),
        "analogy": (
            "选手体力不变，只改允许搜几层楼的规则。"
        ),
        "body": (
            "本实验固定的预训练/微调权重（Shibo-UCSD/coconut-theory）。"
            "训练数据以 ProsQA 3–4 跳题为主，因此模型有「习惯用 3–4 步 latent」的倾向。"
        ),
    },
    "latent-feedback": {
        "title": "Latent 反馈系数",
        "one_liner": (
            "每步把 hidden state 写回 latent 位时的缩放倍数 α。"
        ),
        "analogy": (
            "每层搜楼后，把「情报强度」放大或缩小再传给下一层。"
        ),
        "body": (
            "Coconut 每步 latent 会用上一步最后一层 hidden state 替换 latent token 的嵌入。"
            "第七轮实验在不改题面的前提下，把该向量乘以 α（0.5–2.0）或整体缩放 checkpoint 权重，"
            "观察报边界是否在任务深度附近移动。"
        ),
    },
    "acc-at-depth": {
        "title": "acc@d（步数=跳数准确率）",
        "one_liner": (
            "latent 步数恰好等于题目推理跳数 d 时的准确率。"
        ),
        "analogy": (
            "楼有 5 层就只搜 5 层时，开门命中率是多少。"
        ),
        "body": (
            "比「报边界」更严的对齐指标：边界算法可能因曲线平台或过冲报 6、7 步，"
            "但 acc@d 看的是「步数与题深对齐」时是否真的答对。"
            "第七轮中边界抬高常伴随 acc@d 下降。"
        ),
    },
    "prosqa": {
        "title": "ProsQA",
        "one_liner": (
            "图推理题：判断某节点是否可达，419 道。"
        ),
        "analogy": (
            "给一张关系图，问「A 能到 B 吗」。"
        ),
        "body": (
            "图推理问答数据集：给定有向图与问题，模型需判断某节点是否可达。"
            "本实验主测试集 419 题，推理链只有 3 跳与 4 跳两种，无原生 5–6 跳。"
        ),
    },
    "boundary": {
        "title": "边界",
        "one_liner": (
            "答对率最高的思考步数；并列时取更少步。"
        ),
        "analogy": (
            "开门命中率最高的那一层，不是楼有多高。"
        ),
        "body": (
            "本实验定义的「推荐连续思维步数」：准确率–步数曲线上全局最高点；"
            "若两步准确率相同，取更少步数。不是模型物理上限，而是当前数据下的最划算步数。"
        ),
    },
    "accuracy": {
        "title": "准确率",
        "one_liner": (
            "模型答案与标准答案完全匹配的比例。"
        ),
        "analogy": (
            "十道题里答对几道。"
        ),
        "body": (
            "模型输出与标准答案字符串完全匹配的比例。"
            "全量实验峰值 83.8%（3 步）；5 步降至 74.0%，说明加步过多会有害。"
        ),
    },
    "search-frontier": {
        "title": "搜索前沿（search frontier）",
        "one_liner": (
            "距起点恰好 k 步能摸到的所有节点。"
        ),
        "analogy": (
            "搜救队当前能覆盖到的那一圈楼层。"
        ),
        "body": (
            "BFS 中「恰好距根节点 k 步」的节点集合。"
            "论文发现 latent 向量与 k 跳内可达节点更相似；"
            "第 3 步往往是前沿第一次大幅展开的时刻（跳涨点）。"
        ),
    },
    "reasoning-hops": {
        "title": "推理跳数",
        "one_liner": (
            "标准答案推理链的最短步数（3 或 4 跳）。"
        ),
        "analogy": (
            "题目本身要推理几层关系。"
        ),
        "body": (
            "沿 ground-truth 推理链从根到目标的最短步数（3 跳或 4 跳）。"
            "实验最强规律：分开测时，3 跳题最优 3 步，4 跳题最优 4 步。"
        ),
    },
    "graph-diameter": {
        "title": "图直径",
        "one_liner": (
            "图里最远两点之间的最短距离。"
        ),
        "analogy": (
            "大楼从一端走到另一端最多几层——宽不等于深。"
        ),
        "body": (
            "图中任意两点最短路径的最大值。"
            "实验发现边界跟推理跳数走，不跟直径走："
            "直径≥5 的子集平均跳数仍约 3.7，边界还是 4 步。"
        ),
    },
    "mixed-eval": {
        "title": "混合评估",
        "one_liner": (
            "3 跳题和 4 跳题混在一起测。"
        ),
        "analogy": (
            "三楼四楼住户一起统计开门率。"
        ),
        "body": (
            "3 跳题与 4 跳题混在一起测（全量 419 题）。"
            "3 步 83.8% vs 4 步 83.5% 几乎相同，算法并列取少，故报 3 步而非 4 步。"
        ),
    },
    "saturation": {
        "title": "叠加态饱和",
        "one_liner": (
            "该知道的信息已在 3–4 步编码完，再加步只剩噪声。"
        ),
        "analogy": (
            "人其实就在 3 楼，再搜 6 楼只是重复敲门。"
        ),
        "body": (
            "所需可达性信息已在 3–4 步编码完毕；继续加 latent 不增加新信息，"
            "反而重复扩展搜索前沿、扰动表示，准确率下降。"
        ),
    },
    "synthetic-chain": {
        "title": "人造链（合成链）",
        "one_liner": (
            "人工拼的长链题，和真实题风格差很远。"
        ),
        "analogy": (
            "路牌全外语的假城市——模型没见过。"
        ),
        "body": (
            "实验三/四里手工构造的纯链、稠密链等，token 与拓扑与真实 ProsQA 差异大。"
            "模型几乎不会做（如 5 跳合成链准确率约 5%），此时边界数字无参考价值。"
        ),
    },
    "prosqa-extend": {
        "title": "真实 ProsQA 图延长",
        "one_liner": (
            "在真实 ProsQA 图上把推理链延长到 5–7 跳。"
        ),
        "analogy": (
            "在熟悉的楼里加一层真楼梯，不是搭假楼。"
        ),
        "body": (
            "在原始 ProsQA 图上延长推理链到 5–7 跳，保留原图结构与 token 分布。"
            "5 跳延长可达 5 步边界、96% 准确率——说明分布匹配时边界可以抬高。"
        ),
    },
    "distribution-match": {
        "title": "分布匹配",
        "one_liner": (
            "测试题和训练题在格式、命名上接近。"
        ),
        "analogy": (
            "考题风格和平时练习卷一致，才会做。"
        ),
        "body": (
            "测试题的格式、边表、命名习惯与训练数据接近。"
            "真实延长图可以迁移；纯人造链与训练分布差太远，模型不会做。"
        ),
    },
    "reasoning-superposition-paper": {
        "title": "Reasoning by Superposition（NeurIPS 2025）",
        "one_liner": (
            "提出连续思维+叠加搜索的 NeurIPS 2025 论文。"
        ),
        "analogy": (
            "本实验故事的「理论地图」来源。"
        ),
        "body": (
            "Zhu et al., arXiv:2505.12514。"
            "证明直径为 D 的图可用 D 步连续思维完成可达性搜索；"
            "并分析注意力如何集中在可达边与搜索前沿上。"
        ),
    },
    "correlation": {
        "title": "相关系数 r",
        "one_liner": (
            "边界与平均跳数的相关程度 r=0.543。"
        ),
        "analogy": (
            "题越深，最优思考层数往往也跟着涨——但不是 1:1。"
        ),
        "body": (
            "实验五跨 21 个子集统计：边界与平均推理跳数 r=0.543，与图直径 r=0.501。"
            "说明边界主要由「题要多深」决定，而非固定常数。详见附录 #pattern-laws。"
        ),
    },
    "post4-drop": {
        "title": "post4_drop（4 步后跌幅）",
        "one_liner": (
            "第 5–8 步相对第 4 步的最大跌幅，单位 pp。"
        ),
        "analogy": (
            "第四层已经找对人，第五层起还乱改日志，命中率掉多少。"
        ),
        "body": (
            "实验九核心指标。baseline 全量约 −10.7 pp；"
            "schedule [1,1,1,1,0,0,0,0] 可压到约 −1.0 pp，acc@4 不变。"
            "详见附录 #feedback-schedule 与 #post4-playbook。"
        ),
    },
    "auto-route": {
        "title": "auto_route（按题 BFS 路由）",
        "one_liner": (
            "无标签通解：n_latent = 题面 BFS 深度 d。"
        ),
        "analogy": (
            "先量目标在第几层，再派搜救队搜几层——不是全队统一只搜 3 层。"
        ),
        "body": (
            "实验十：d = max(BFS(root→候选1), BFS(root→候选2))，令 n_latent = d。"
            "全量 419 题 93.1%，比 fixed_3 高 9.3 pp，且等于结构金标准 oracle_hop。"
            "详见附录 #auto-submit 与实验室 #exp10Panel。"
        ),
    },
    "latent-exp": {
        "title": "000003 Latent Token 实验",
        "one_liner": (
            "四楼项目：研究未揭晓格要开几个。"
        ),
        "analogy": (
            "隔壁实验问宽度；本实验问深度。"
        ),
        "body": (
            "四楼项目「未揭晓的格子」：研究扩散模型 Mask 里要放几个 latent token 才够用。"
            "曲线通常单调上升到平台（够用点），与 000002 的峰值边界是两种不同规律。"
        ),
    },
    "trainable-stop-feasible": {
        "title": "训练自停可行（trainable_stop_feasible）",
        "one_liner": (
            "test 答题 ≥ fixed_3 且停步时机 ≥ 50% 才算可行。"
        ),
        "analogy": (
            "既要找对人，又要在对的楼层收队——两个指标都要达标。"
        ),
        "body": (
            "实验十一至五十五的 CPU 判定：main_strategy_accuracy ≥ fixed_3（86.3% on test 168）"
            "且 stop_timing_acc ≥ 50%。"
            "GPU 定稿后分层评价：L1 deployable_mvp（acc+算力+无 oracle）已达成；"
            "L2 strict_feasible（+timing≥50%）未过，全量最好约 37%。"
            "teacher 线（28/29）用 oracle fc 标签可达 ≈95% 且 timing 可行，非盲部署。"
        ),
    },
    "stop-timing-acc": {
        "title": "停步时机准确率（stop_timing_acc）",
        "one_liner": (
            "停步步数是否等于「首次答对步」的比例。"
        ),
        "analogy": (
            "收队时是否刚好在目标楼层，而不是早退或拖堂。"
        ),
        "body": (
            "自停实验核心指标：模型停步 n 是否等于 first_correct（fc）。"
            "L2 strict_feasible 要求 timing ≥ 50%；当前全量最好约 37%，test 约 39%。"
            "min_n=3 理论上限约 44%；失败主因 late_stop（≈57%），非 early_stop（≈5%）。"
            "前缀线答题率可达 94% 但 timing 约 28–30%——混测 3/4 跳混淆加剧 timing 损失。"
        ),
    },
    "stop-head": {
        "title": "Stop Head（停步头）",
        "one_liner": (
            "挂在 Coconut hidden 上的小网络，预测「这一步该不该停」。"
        ),
        "analogy": (
            "搜救队长每上一层看一眼仪表盘，决定继续还是收队。"
        ),
        "body": (
            "实验十一训 LatentStopHead；实验十三起 RichStopHead 加入 step、答案桶、稳定 streak 等特征。"
            "冻结 Coconut 只训 head 时 test 表现差；实验十五联合微调略有提升但仍未可行。"
            "实验五十三至五十五在线组合 head∨稳定∨收敛。"
        ),
    },
    "upfront-budget": {
        "title": "Upfront 预算（前缀自报步数）",
        "one_liner": (
            "推理前先决定 n_latent，多数只需 1–2 次 forward。"
        ),
        "analogy": (
            "出发前就定好搜几层，而不是搜到一半再改计划。"
        ),
        "body": (
            "实验三十一至五十二：用 Coconut 前缀、图特征 Δ、kNN 集成等在 test 前预测 n∈{3,4}。"
            "答题率平台约 94.0%，高于 auto_route 92.9%，但 stop_timing_acc 约 28–30%，未达可行。"
            "说明混测 3/4 跳混淆是 timing 瓶颈，而非 forward 次数。"
        ),
    },
    "online-self-stop": {
        "title": "在线自停（逐步 OR 组合）",
        "one_liner": (
            "逐步 latent 推理中，head∨稳定∨收敛任一触发即停。"
        ),
        "analogy": (
            "边搜边听三个信号：仪表盘、答案是否稳住、队员是否还在动。"
        ),
        "body": (
            "实验五十三至五十五：不 upfront 预算、不用 BFS 路由，"
            "在逐步推理中用 RichStopHead、答案稳定 streak、hidden 收敛等 OR 组合决定停步。"
            "与实验十 BFS 路由对照：一个边搜边停，一个先量深度再搜。"
        ),
    },
    "oracle-first-correct": {
        "title": "Oracle 首次答对步（first_correct）",
        "one_liner": (
            "若某步首次答对，理想停步就是该步——teacher 上界标签。"
        ),
        "analogy": (
            "事后看录像，第一次敲对门是第几层——部署时拿不到这录像。"
        ),
        "body": (
            "自停实验的 stop 标签与上界：oracle_first_correct_accuracy 约 97.6%。"
            "实验 28/29 等用该标签或变体（soft floor）训练/校准，可达 ≈95% 且 feasible，"
            "但需要知道答案何时首次正确，属 teacher 上界而非盲部署。"
            "Phase 19 U5 失败解剖：56% 题 fc 在第 1–2 步，min_n=3 下这些题 timing 永远算 late_stop。"
        ),
    },
    "deployable-mvp": {
        "title": "deployable_mvp（可部署 MVP）",
        "one_liner": (
            "答对率与算力达标、推理无 oracle——「够好就停」能上线的标准。"
        ),
        "analogy": (
            "搜救队自己判断收队、不偷看标准答案、平均搜层数也省——任务能交差。"
        ),
        "body": (
            "L1 评价标准：全量 acc ≥ fixed_3（86.3%）、mean_stop_n ≤ 4.5、推理过程无 oracle 扫步，"
            "且 M3 多走步有害已证实（搞砸 24.4%、救回 0%）。"
            "Phase 17 定稿：deployable_mvp=True；推荐 knn_min3_full，全量 acc 92.6%，mean_n=3.37。"
            "timing≥50% 是 stretch goal（L2 strict_feasible），当前未过。"
        ),
    },
    "strict-feasible": {
        "title": "strict_feasible（严格可行）",
        "one_liner": (
            "deployable_mvp 基础上，还要求 timing≥50%。"
        ),
        "analogy": (
            "不但找对人，还要一半以上的题刚好在目标楼层收队。"
        ),
        "body": (
            "L2 评价标准：在 deployable_mvp 全部条件之上，另需 stop_timing_acc ≥ 50%。"
            "当前未达标：全量最好约 37%（min_n=3, thr=0.35），test 子集约 39%，"
            "距 50% 差 11–13 个百分点。min_n=3 理论 timing 上限约 44%。"
        ),
    },
    "min-n": {
        "title": "min_n（最少步数下界）",
        "one_liner": (
            "至少要想满 n 步才能停——保准确率，也卡 timing。"
        ),
        "analogy": (
            "规定「至少搜满 3 层才能收队」——很多 1、2 层就找对人了，计分却算停晚。"
        ),
        "body": (
            "推理时下界约束：就算 stop head 很想在第 1、2 步停，min_n=3 也不允许。"
            "min_n 越大 acc 越稳、timing 越低。当前部署推荐 min_n=3。"
            "56% 题首次答对在第 1–2 步，min_n=3 下这些题 timing 不可能命中。"
            "Phase 20 计划从 min_n=2 侧突破（理论 timing 上限约 61%）。"
        ),
    },
    "late-stop": {
        "title": "late_stop（停太晚）",
        "one_liner": (
            "首次答对步已过，还被规则拖着不能早停。"
        ),
        "analogy": (
            "第 2 层就找到人了，规定必须搜到第 3 层——计分算你停晚了。"
        ),
        "body": (
            "Phase 19 U5 失败解剖三类之一：模型停步 n > first_correct（fc）。"
            "在 min_n=3 配置下占约 57%（phase16_best_timing）；early_stop 仅约 5%。"
            "主因是 min_n=3 不让 fc=1/2 的题早停，而非模型太急着想停。"
        ),
    },
    "early-stop": {
        "title": "early_stop（停太早）",
        "one_liner": (
            "在首次能答对之前就停了。"
        ),
        "analogy": (
            "目标在第 3 层，搜救队第 1 层就收队——停早了。"
        ),
        "body": (
            "Phase 19 U5 失败解剖：停步 n < first_correct。"
            "在 min_n=3 最优配置下仅占约 5%——不是当前 timing 瓶颈。"
            "若 early_stop 主导，应对 patience/延后停步；本实验 late_stop 主导。"
        ),
    },
    "first-correct": {
        "title": "first_correct（fc，首次答对步）",
        "one_liner": (
            "从第 1 步起，第一次输出正确答案的 latent 步数。"
        ),
        "analogy": (
            "录像里第一次敲对门是第几层——timing 评分要对齐这一层。"
        ),
        "body": (
            "timing 指标的 ground truth：模型应在 fc 步停才算「停得准」。"
            "ProsQA 419 题中 408 题有 fc；分布：fc=1 约 160 题、fc=2 约 69 题、fc=3 约 156 题。"
            "fc<3 的题在 min_n=3 下不可能 timing 命中——这是算术瓶颈。"
        ),
    },
    "knn-min3": {
        "title": "knn_min3_full（推荐部署方案）",
        "one_liner": (
            "M2 stop head + kNN 前缀 floor + min_n=3，全量 acc 92.6%。"
        ),
        "analogy": (
            "队长自己判断停，出发前用邻居题的经验校正「至少搜几层」。"
        ),
        "body": (
            "Phase 16–17 GPU 定稿推荐方案：global_min_n=3，threshold=0.15，kNN floor 校正。"
            "全量 419 题 acc 92.6%，mean_n=3.37，timing 29.4%，无 oracle，deployable_mvp=True。"
            "比纯 M2 head（87.5%）高，比 hybrid 上界（97%，含 BFS）规矩干净。"
        ),
    },
    "hybrid-stop": {
        "title": "hybrid 停步（上界参照）",
        "one_liner": (
            "先 BFS 估深度再序贯扫 fc——acc 可达 97%，但含 oracle 信息。"
        ),
        "analogy": (
            "先打听了楼层再搜——分数好看，但不算公平比赛的盲停。"
        ),
        "body": (
            "Phase 6–9 GPU 验证：hybrid 全量 acc 约 96–97%，多 seed 稳定。"
            "含 BFS 深度估计与序贯扫步，非 deployable_mvp。"
            "与 M2+knn 盲停（92.6%）对照：hybrid 作性能上界，M2+knn 作部署线。"
            "Phase 19 U3 learned hybrid acc 95.7% 但 timing 22.8%。"
        ),
    },
    "m2-head": {
        "title": "M2 head（RichStopHead v3）",
        "one_liner": (
            "实验十三 RichStopHead——后续 GPU 实验的基线停步头。"
        ),
        "analogy": (
            "带丰富仪表盘的搜救队长——看隐状态、步数、答案桶、稳定 streak。"
        ),
        "body": (
            "实验十三训练 RichStopHead（答案桶+稳定 streak+focal loss），冻结 Coconut。"
            "test 168 题：acc 87.5%，timing 34.8%——后续 Phase 13–19 反复对照的标杆。"
            "全量 419 题配合 min_n=3 网格可达 timing 37%；配合 knn floor 可达 acc 92.6%。"
        ),
    },
    "timing-ceiling": {
        "title": "timing 天花板",
        "one_liner": (
            "min_n=3 下理论 timing 上限约 44%；实测 37% 已达 84%。"
        ),
        "analogy": (
            "一半题规定必须搜满 3 层——就算停步完美，timing 也过不了半。"
        ),
        "body": (
            "CPU 分析 timing_ceiling_analysis.json：408 题有 fc，229 题（56%）fc<3。"
            "min_n=3 理论上限 ceilings[3]≈43.9%；当前最优 37.0% 达上限 84%。"
            "min_n=2 理论上限约 60.8%——Phase 20 从此侧突破。"
            "Phase 13–19 换标签、联合长训均未破——瓶颈在规则与 fc 分布，非训练不足。"
        ),
    },
}


def esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def render_text_with_terms(text: str) -> str:
    """把 [[id|标签]] 转成指向 glossary 的链接，其余部分转义。"""
    if not text:
        return ""
    parts: list[str] = []
    last = 0
    for m in TERM_PATTERN.finditer(text):
        parts.append(esc(text[last : m.start()]))
        tid, label = m.group(1), m.group(2)
        if tid in TERMS:
            parts.append(
                f'<a class="term-ref" href="{GLOSSARY_URL}#{esc(tid)}" '
                f'title="查看注释：{esc(TERMS[tid]["title"])}" '
                f'aria-label="术语注释：{esc(label)}">{esc(label)}</a>'
            )
        else:
            parts.append(esc(label))
        last = m.end()
    parts.append(esc(text[last:]))
    return "".join(parts)


def _render_glossary_entry(tid: str, entry: dict) -> str:
    parts = [
        f'<article class="glossary-entry" id="{esc(tid)}">',
        f'<h2>{esc(entry["title"])}</h2>',
    ]
    if entry.get("one_liner"):
        parts.append(f'<p class="glossary-one-liner">{esc(entry["one_liner"])}</p>')
    if entry.get("analogy"):
        parts.append(f'<p class="glossary-analogy">打个比方：{esc(entry["analogy"])}</p>')
    parts.append('<p class="glossary-detail-label">展开说明</p>')
    parts.append(f'<p>{esc(entry["body"])}</p>')
    parts.append(f'<a class="glossary-back-top" href="#top">↑ 回到页首</a></article>')
    return "".join(parts)


EXPERIMENT_INDEX = [
    ("实验一 · 全量扫描", "index.html#ch-how1", "lab.html#exp1Panel"),
    ("实验二 · 多数据集", "index.html#ch-how2", "lab.html#exp2Panel"),
    ("实验三 · 深边界", "index.html#ch-how3", "lab.html#exp3Panel"),
    ("实验四 · 构造对照", "index.html#ch-how3", "lab.html#exp4Panel"),
    ("实验五 · 规律寻探", "index.html#ch-how4", "appendix.html#pattern-laws"),
    ("实验六 · 边界上推", "index.html#ch-how5", "lab.html#exp6Panel"),
    ("实验七 · 模型扰动", "index.html#ch-how6", "lab.html#exp7Panel"),
    ("实验八 · 7–8 步上推", "index.html#ch-how7", "lab.html#exp8Panel"),
    ("实验九 · 停写回", "index.html#ch-how8", "lab.html#exp9Panel"),
    ("实验十 · 通解", "index.html#ch-how10", "lab.html#exp10Panel"),
    ("实验十一–五十五 · 自停", "index.html#ch-adaptive-11", "appendix.html#adaptive-stop"),
    ("GPU Phase 16–17 · 定稿", "index.html#ch-gpu-16-17", "appendix.html#gpu-phase"),
    ("GPU Phase 18–19 · timing", "index.html#ch-gpu-18-19", "appendix.html#gpu-phase"),
    ("GPU Phase 20 · 待验证", "index.html#ch-gpu-20", "lab.html#labCatGpu"),
]


def render_glossary_page() -> str:
    items = [_render_glossary_entry(tid, entry) for tid, entry in TERMS.items()]
    nav = "".join(
        f'<a href="#{esc(tid)}">{esc(entry["title"])}</a>'
        for tid, entry in TERMS.items()
    )
    exp_rows = "".join(
        f'<tr><td>{esc(title)}</td>'
        f'<td><a href="{esc(story)}">故事</a></td>'
        f'<td><a href="{esc(data)}">数据/实验</a></td></tr>'
        for title, story, data in EXPERIMENT_INDEX
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>术语注释 — 三楼与四楼之间</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=ZCOOL+KuaiLe&family=Noto+Sans+SC:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/css/site-theme.css">
  <link rel="stylesheet" href="/css/responsive.css">
  <link rel="stylesheet" href="/css/theme/experiment.css">
  <link rel="stylesheet" href="/css/theme/youth-light.css">
</head>
<body class="theme-youth glossary-body exp-page-glossary">
  <div class="page glossary-page" id="top">
    <header>
      <a href="index.html" class="back-link">← 回到实验故事</a>
      <h1>术语注释</h1>
      <p class="subtitle">每条术语先给「一句话」和「打个比方」，再展开细节。</p>
      <nav class="exp-page-nav" aria-label="实验导航">
        <a class="exp-nav-btn secondary findings-refresh-btn findings-jump-btn" href="index.html" data-exp-nav="story">实验故事</a>
        <a class="exp-nav-btn secondary findings-refresh-btn findings-jump-btn" href="glossary.html" data-exp-nav="glossary" aria-current="page">术语注释</a>
        <a class="exp-nav-btn secondary findings-refresh-btn findings-jump-btn" href="appendix.html" data-exp-nav="appendix">科学附录</a>
        <a class="exp-nav-btn secondary findings-refresh-btn findings-jump-btn" href="lab.html" data-exp-nav="lab">交互复现实验</a>
      </nav>
    </header>
    <section class="glossary-exp-index">
      <h2>实验索引（1–10 + 自停 + GPU）</h2>
      <p class="muted">每轮链接到故事章节与实验室/附录数据；GPU Phase 见科学附录 #gpu-phase。</p>
      <table class="compare-table">
        <thead><tr><th>实验</th><th>故事</th><th>数据 / 复现</th></tr></thead>
        <tbody>{exp_rows}</tbody>
      </table>
    </section>
    <nav class="glossary-nav" aria-label="术语目录">
      <h2>目录</h2>
      <div class="glossary-nav-links">{nav}</div>
    </nav>
    <section class="glossary-list">
      {"".join(items)}
    </section>
    <footer class="glossary-foot">
      <a href="index.html#story-guide">返回《三楼与四楼之间》</a>
    </footer>
  </div>
</body>
</html>
"""


def write_glossary_html() -> Path:
    GLOSSARY_HTML.write_text(render_glossary_page(), encoding="utf-8")
    return GLOSSARY_HTML
