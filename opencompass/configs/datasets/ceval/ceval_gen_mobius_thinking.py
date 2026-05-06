from opencompass.datasets import CEvalDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever


ceval_subject_mapping = {
    'computer_network': ['Computer Network', '计算机网络', 'STEM'],
    'operating_system': ['Operating System', '操作系统', 'STEM'],
    'computer_architecture': ['Computer Architecture', '计算机组成', 'STEM'],
    'college_programming': ['College Programming', '大学编程', 'STEM'],
    'college_physics': ['College Physics', '大学物理', 'STEM'],
    'college_chemistry': ['College Chemistry', '大学化学', 'STEM'],
    'advanced_mathematics': ['Advanced Mathematics', '高等数学', 'STEM'],
    'probability_and_statistics': ['Probability and Statistics', '概率统计', 'STEM'],
    'discrete_mathematics': ['Discrete Mathematics', '离散数学', 'STEM'],
    'electrical_engineer': ['Electrical Engineer', '注册电气工程师', 'STEM'],
    'metrology_engineer': ['Metrology Engineer', '注册计量师', 'STEM'],
    'high_school_mathematics': ['High School Mathematics', '高中数学', 'STEM'],
    'high_school_physics': ['High School Physics', '高中物理', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', '高中化学', 'STEM'],
    'high_school_biology': ['High School Biology', '高中生物', 'STEM'],
    'middle_school_mathematics': ['Middle School Mathematics', '初中数学', 'STEM'],
    'middle_school_biology': ['Middle School Biology', '初中生物', 'STEM'],
    'middle_school_physics': ['Middle School Physics', '初中物理', 'STEM'],
    'middle_school_chemistry': ['Middle School Chemistry', '初中化学', 'STEM'],
    'veterinary_medicine': ['Veterinary Medicine', '兽医学', 'STEM'],
    'college_economics': ['College Economics', '大学经济学', 'Social Science'],
    'business_administration': ['Business Administration', '工商管理', 'Social Science'],
    'marxism': ['Marxism', '马克思主义基本原理', 'Social Science'],
    'mao_zedong_thought': ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science'],
    'education_science': ['Education Science', '教育学', 'Social Science'],
    'teacher_qualification': ['Teacher Qualification', '教师资格', 'Social Science'],
    'high_school_politics': ['High School Politics', '高中政治', 'Social Science'],
    'high_school_geography': ['High School Geography', '高中地理', 'Social Science'],
    'middle_school_politics': ['Middle School Politics', '初中政治', 'Social Science'],
    'middle_school_geography': ['Middle School Geography', '初中地理', 'Social Science'],
    'modern_chinese_history': ['Modern Chinese History', '近代史纲要', 'Humanities'],
    'ideological_and_moral_cultivation': ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities'],
    'logic': ['Logic', '逻辑学', 'Humanities'],
    'law': ['Law', '法学', 'Humanities'],
    'chinese_language_and_literature': ['Chinese Language and Literature', '中国语言文学', 'Humanities'],
    'art_studies': ['Art Studies', '艺术学', 'Humanities'],
    'professional_tour_guide': ['Professional Tour Guide', '导游资格', 'Humanities'],
    'legal_professional': ['Legal Professional', '法律职业资格', 'Humanities'],
    'high_school_chinese': ['High School Chinese', '高中语文', 'Humanities'],
    'high_school_history': ['High School History', '高中历史', 'Humanities'],
    'middle_school_history': ['Middle School History', '初中历史', 'Humanities'],
    'civil_servant': ['Civil Servant', '公务员', 'Other'],
    'sports_science': ['Sports Science', '体育学', 'Other'],
    'plant_protection': ['Plant Protection', '植物保护', 'Other'],
    'basic_medicine': ['Basic Medicine', '基础医学', 'Other'],
    'clinical_medicine': ['Clinical Medicine', '临床医学', 'Other'],
    'urban_and_rural_planner': ['Urban and Rural Planner', '注册城乡规划师', 'Other'],
    'accountant': ['Accountant', '注册会计师', 'Other'],
    'fire_engineer': ['Fire Engineer', '注册消防工程师', 'Other'],
    'environmental_impact_assessment_engineer': ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other'],
    'tax_accountant': ['Tax Accountant', '税务师', 'Other'],
    'physician': ['Physician', '医师资格', 'Other'],
}
ceval_all_sets = list(ceval_subject_mapping.keys())


def make_example(question: str, A: str, B: str, C: str, D: str, answer: str,
                 think: str) -> dict:
    return dict(
        question=question,
        options=dict(A=A, B=B, C=C, D=D),
        answer=answer,
        think=think,
    )


manual_fewshot_examples = {
    'computer_network': [
        make_example(
            '滑动窗口的作用是____。',
            '流量控制',
            '拥塞控制',
            '路由控制',
            '差错控制',
            'A',
            '题目问的是滑动窗口机制本身的主要作用。滑动窗口通过限制发送方未确认数据的数量，使发送速率与接收方处理能力匹配，所以它解决的是流量控制问题。拥塞控制关注整个网络负载，路由控制关注路径选择，差错控制主要依赖校验和重传。由此可知最终答案是A。',
        )
    ],
    'operating_system': [
        make_example(
            '在时间片轮转调度算法中，正在运行的进程时间片用完后将进入____状态。',
            '阻塞',
            '就绪',
            '结束',
            '挂起',
            'B',
            '时间片轮转调度里，进程时间片用完并不意味着它无法继续执行，只是本轮CPU使用结束，需要重新排队等待下一次调度。因此它会从运行态回到就绪态，而不是进入阻塞态。结束和挂起都不符合题意，所以最终答案是B。',
        )
    ],
    'computer_architecture': [
        make_example(
            '计算机中负责执行算术运算和逻辑运算的部件是____。',
            '控制器',
            '运算器',
            '存储器',
            '输入设备',
            'B',
            '题目考查计算机硬件功能划分。算术运算和逻辑运算由ALU所在的运算器完成；控制器负责发出控制信号，存储器负责保存数据，输入设备负责输入信息。因此真正承担运算任务的是运算器，最终答案是B。',
        )
    ],
    'college_programming': [
        make_example(
            '在C语言中，长度为5的数组 `int a[5];` 的合法下标范围是____。',
            '1到5',
            '0到4',
            '-1到4',
            '0到5',
            'B',
            'C语言数组下标从0开始，长度为5意味着一共有5个元素，对应的位置依次是0、1、2、3、4。下标5已经越界，下标-1也不合法。所以合法范围只能是0到4，最终答案是B。',
        )
    ],
    'college_physics': [
        make_example(
            '有一劲度系数为k的轻弹簧，竖直放置，下端悬一质量为m的小球，开始时使弹簧为原长而小球恰好与地接触，今将弹簧上端缓慢地提起，直到小球刚能脱离地面为止，在此过程中外力作功为____',
            'm^2g^2/(4k)',
            'm^2g^2/(3k)',
            'm^2g^2/(2k)',
            '2m^2g^2/k',
            'C',
            '缓慢提起说明过程可按准静态处理。小球脱离地面前始终未上升，所以外力做功全部转化为弹簧弹性势能。设上端上移x，则弹簧拉力为kx；当小球刚脱离地面时支持力为0，故kx=mg，得x=mg/k。于是外力做功 W=∫(0到mg/k)kx dx=1/2·k·(mg/k)^2=m^2g^2/(2k)。因此最终答案是C。',
        )
    ],
    'college_chemistry': [
        make_example(
            '在25摄氏度时，pH小于7的溶液通常呈____。',
            '酸性',
            '碱性',
            '中性',
            '无法判断',
            'A',
            '25摄氏度时纯水的pH约为7，对应中性。若某溶液pH小于7，说明其中氢离子浓度高于纯水，因此呈酸性。碱性溶液应当是pH大于7，所以最终答案是A。',
        )
    ],
    'advanced_mathematics': [
        make_example(
            '函数 sinx 的导数是____。',
            'sinx',
            'cosx',
            '-sinx',
            '-cosx',
            'B',
            '这是高等数学中最基本的导数公式之一。常见三角函数导数里，(sinx)\'=cosx，(cosx)\'=-sinx。只要把标准公式对应好，就能排除其他选项，因此最终答案是B。',
        )
    ],
    'probability_and_statistics': [
        make_example(
            '若事件A与事件B互斥，则 P(A∩B) = ____。',
            '0',
            '1',
            'P(A)+P(B)',
            'P(A)P(B)',
            'A',
            '互斥表示两个事件不能同时发生，所以A与B的交集是不可能事件。不可能事件的概率等于0，因此 P(A∩B)=0。其余选项分别对应别的概率关系，不符合互斥定义，所以最终答案是A。',
        )
    ],
    'discrete_mathematics': [
        make_example(
            '命题“所有学生都通过考试”的否定是____。',
            '所有学生都没通过考试',
            '至少有一名学生没有通过考试',
            '至少有一名学生通过考试',
            '没有学生参加考试',
            'B',
            '全称命题“所有……都……”的否定，应改成存在命题“至少有一个……不……”。所以“所有学生都通过考试”的否定不是“所有学生都没通过”，而是“至少有一名学生没有通过考试”。因此最终答案是B。',
        )
    ],
    'electrical_engineer': [
        make_example(
            '测量某元件两端电压时，电压表应当____连接。',
            '串联在电路中',
            '并联在元件两端',
            '接在电源外部即可',
            '只接一根导线即可',
            'B',
            '电压反映的是两点之间的电势差，所以测量时必须把电压表并联在被测元件两端。串联会改变原电路并且不是标准测压方式，后两项更不成立。因此最终答案是B。',
        )
    ],
    'metrology_engineer': [
        make_example(
            '对同一量进行多次重复测量并取平均值，主要是为了减小____。',
            '随机误差',
            '系统误差',
            '粗大误差',
            '单位换算误差',
            'A',
            '重复测量取平均的核心作用，是让正负波动相互抵消，从而减小随机误差的影响。系统误差不会因为简单求平均自动消失，粗大误差更需要剔除而不是平均。单位换算问题与重复测量无关，因此最终答案是A。',
        )
    ],
    'high_school_mathematics': [
        make_example(
            '方程 2x - 3 = 5 的解是____。',
            '1',
            '2',
            '3',
            '4',
            'D',
            '解一元一次方程时先移项：2x=8，再两边同时除以2，得到x=4。只要按代数步骤计算，就能确定其余选项都不对。因此最终答案是D。',
        )
    ],
    'high_school_physics': [
        make_example(
            '忽略空气阻力时，自由落体运动的加速度大小为____。',
            '0',
            'g',
            '2g',
            '随高度变化无法确定',
            'B',
            '自由落体在忽略空气阻力时，只受重力作用，因此其加速度就是重力加速度g。题目问的是经典理想模型下的结论，不需要引入其他复杂因素，所以最终答案是B。',
        )
    ],
    'high_school_chemistry': [
        make_example(
            '在H2O分子中，氧元素的化合价通常为____。',
            '-2',
            '-1',
            '+1',
            '+2',
            'A',
            '水分子整体化合价代数和为0，两个氢通常各为+1，总和是+2，因此氧必须是-2才能平衡。-1常见于过氧化物，不适用于普通的H2O，所以最终答案是A。',
        )
    ],
    'high_school_biology': [
        make_example(
            'DNA的基本组成单位是____。',
            '氨基酸',
            '葡萄糖',
            '核苷酸',
            '脂肪酸',
            'C',
            'DNA属于核酸，核酸的基本组成单位是核苷酸。氨基酸是蛋白质的基本单位，葡萄糖属于糖类，脂肪酸是脂质相关成分，都不对题。因此最终答案是C。',
        )
    ],
    'middle_school_mathematics': [
        make_example(
            '若 3x = 12，则 x = ____。',
            '2',
            '3',
            '4',
            '6',
            'C',
            '由3x=12可知，x等于12除以3，因此x=4。其余选项代回原式都不能成立，所以最终答案是C。',
        )
    ],
    'middle_school_biology': [
        make_example(
            '植物进行光合作用的主要场所是____。',
            '细胞核',
            '叶绿体',
            '液泡',
            '细胞壁',
            'B',
            '光合作用需要叶绿素和相关酶体系，主要发生在叶绿体中。细胞核负责遗传信息调控，液泡和细胞壁都不是光合作用的主要场所。因此最终答案是B。',
        )
    ],
    'middle_school_physics': [
        make_example(
            '物体密度的计算公式是____。',
            'ρ = m/V',
            'ρ = V/m',
            'ρ = mg',
            'ρ = m+V',
            'A',
            '密度定义为单位体积内的质量，所以公式是ρ=m/V。V/m在量纲上就不对，mg是重力，m+V更没有物理意义。因此最终答案是A。',
        )
    ],
    'middle_school_chemistry': [
        make_example(
            '下列物质中，属于化合物的是____。',
            '氧气',
            '铁',
            '水',
            '空气',
            'C',
            '化合物由不同元素组成且具有固定组成。水由氢和氧两种元素组成，属于典型化合物；氧气和铁都是单质，空气是混合物。因此最终答案是C。',
        )
    ],
    'veterinary_medicine': [
        make_example(
            '狂犬病的病原体属于____。',
            '病毒',
            '细菌',
            '真菌',
            '寄生虫',
            'A',
            '狂犬病是由狂犬病病毒引起的传染病，核心识别点就是其病原体类别。细菌、真菌和寄生虫都不符合这一疾病的基础病原学知识，因此最终答案是A。',
        )
    ],
    'college_economics': [
        make_example(
            '宏观经济政策的常见目标通常不包括____。',
            '充分就业',
            '物价稳定',
            '经济增长',
            '完全消除市场竞争',
            'D',
            '宏观经济政策常见目标包括促进增长、稳定物价、实现较高就业和保持国际收支平衡。A、B、C都属于教材中的标准表述。D所说“完全消除市场竞争”既不是宏观调控目标，也违背市场经济基本逻辑，因此最终答案是D。',
        )
    ],
    'business_administration': [
        make_example(
            '管理的基本职能通常不包括____。',
            '计划',
            '组织',
            '领导',
            '随机应变而不做控制',
            'D',
            '经典管理职能一般包括计划、组织、领导和控制。A、B、C都在这个框架中，而D把“不做控制”当作职能，显然违背管理学基本概念。因此最终答案是D。',
        )
    ],
    'marxism': [
        make_example(
            '唯物辩证法的总特征是____。',
            '静止和孤立',
            '普遍联系和永恒发展',
            '绝对平均和完全平衡',
            '主观想象决定客观现实',
            'B',
            '唯物辩证法强调世界是普遍联系、不断发展的。A与辩证法相反，C不是其基本原理，D则属于唯心主义倾向。因此能准确概括总特征的只有B，最终答案是B。',
        )
    ],
    'mao_zedong_thought': [
        make_example(
            '在新民主主义革命中，领导力量是____。',
            '民族资产阶级',
            '中国共产党',
            '农民阶级',
            '城市小资产阶级',
            'B',
            '新民主主义革命涉及多个革命阶级参与，但领导力量必须和主力军区分开来。毛泽东思想明确指出，中国共产党是中国革命的领导核心。农民是主力军之一，但不是领导力量，因此最终答案是B。',
        )
    ],
    'education_science': [
        make_example(
            '教育活动的直接目的在于____。',
            '促进人的发展',
            '扩大社会消费',
            '增加财政收入',
            '提高出口规模',
            'A',
            '题干问的是教育活动的直接目的，要抓住“直接”二字。教育首先面向人的成长，通过知识、能力和品德培养促进个体发展。B、C、D都可能是社会层面的间接效应，但不是教育活动本身的直接目标，因此最终答案是A。',
        )
    ],
    'teacher_qualification': [
        make_example(
            '教师劳动的对象主要是____。',
            '学生',
            '教材',
            '教室',
            '考试分数',
            'A',
            '教师工作的核心对象是学生，教材、教室和分数都只是教育活动中的工具、环境或结果。理解教师职业特性时，首先要抓住“育人”这一根本，因此最终答案是A。',
        )
    ],
    'high_school_politics': [
        make_example(
            '商品的二因素是____。',
            '使用价值和价值',
            '价格和成本',
            '供给和需求',
            '货币和交换',
            'A',
            '政治经济学中商品具有两个基本属性，即使用价值和价值。价格是价值的货币表现，不是商品的二因素本身；供给需求描述市场关系，也不对题。因此最终答案是A。',
        )
    ],
    'high_school_geography': [
        make_example(
            '地球自转的方向是____。',
            '自西向东',
            '自东向西',
            '自南向北',
            '自北向南',
            'A',
            '地球自转的标准表述是自西向东。从北极上空看呈逆时针，从南极上空看呈顺时针，但本质方向不变。因此最终答案是A。',
        )
    ],
    'middle_school_politics': [
        make_example(
            '我国的根本政治制度是____。',
            '人民代表大会制度',
            '民族区域自治制度',
            '基层群众自治制度',
            '中国共产党领导的多党合作制度',
            'A',
            '题目问“根本政治制度”，这是政治常识中的固定表述。人民代表大会制度是我国的根本政治制度，后面几个是重要制度但不是“根本政治制度”这一概念。因此最终答案是A。',
        )
    ],
    'middle_school_geography': [
        make_example(
            '在没有特别说明的普通地图上，通常采用的方向判断规则是____。',
            '上北下南，左西右东',
            '上南下北，左东右西',
            '上东下西，左北右南',
            '上西下东，左南右北',
            'A',
            '普通地图默认遵循“上北下南，左西右东”的判向规则，这是最基础的地图知识。只有在有指向标或经纬网特殊说明时，才需要按其他方法判断。因此最终答案是A。',
        )
    ],
    'modern_chinese_history': [
        make_example(
            '五四运动爆发于____年。',
            '1919',
            '1921',
            '1937',
            '1949',
            'A',
            '五四运动是中国近代史上的重要事件，爆发时间是1919年。1921年对应中国共产党成立，1937年对应全面抗战开始，1949年对应新中国成立。因此最终答案是A。',
        )
    ],
    'ideological_and_moral_cultivation': [
        make_example(
            '爱国主义的核心是____。',
            '爱国',
            '求新',
            '逐利',
            '排外',
            'A',
            '题目考查概念本身。爱国主义首先强调对国家、民族和人民的深厚认同与责任，核心当然是“爱国”。其余选项要么偏离主题，要么带有错误价值导向，因此最终答案是A。',
        )
    ],
    'logic': [
        make_example(
            '“如果P则Q；非Q；所以非P”属于____。',
            '否定后件式',
            '肯定后件式',
            '否定前件式',
            '二难推理',
            'A',
            '这个推理形式是经典的有效推理：由P→Q和非Q，可以推出非P，逻辑学中称为否定后件式。肯定后件和否定前件都不是这个结构，因此最终答案是A。',
        )
    ],
    'law': [
        make_example(
            '下列选项中，不属于民法基本原则的是____。',
            '自愿原则',
            '公平原则',
            '诚实信用原则',
            '有罪推定原则',
            'D',
            '民法的基本原则包括平等、自愿、公平、诚信、公序良俗等。A、B、C都属于民法核心原则。D“有罪推定”既不属于民法，也与现代法治强调的无罪推定精神相悖，所以最终答案是D。',
        )
    ],
    'chinese_language_and_literature': [
        make_example(
            '《红楼梦》的作者是____。',
            '曹雪芹',
            '施耐庵',
            '罗贯中',
            '吴承恩',
            'A',
            '中国古典四大名著作者要准确对应：《红楼梦》是曹雪芹所著，《水浒传》对应施耐庵，《三国演义》对应罗贯中，《西游记》对应吴承恩。因此最终答案是A。',
        )
    ],
    'art_studies': [
        make_example(
            '电影通常被归类为____。',
            '综合艺术',
            '纯语言艺术',
            '纯造型艺术',
            '纯表情艺术',
            'A',
            '电影同时综合了文学、表演、音乐、摄影、美术等多种艺术因素，因此在艺术学中通常归为综合艺术。若把它看成单一语言或单一造型艺术，都过于狭窄，所以最终答案是A。',
        )
    ],
    'professional_tour_guide': [
        make_example(
            '地陪导游首次接团时，首先应核实的是____。',
            '旅游团和行程信息',
            '游客购物预算',
            '游客私人联系方式',
            '景区门票收藏价值',
            'A',
            '导游首次接团最重要的是确保接待对象和接待安排准确无误，所以应优先核实团队身份、人数和行程信息。购物预算和收藏价值都不是接团首要事项，因此最终答案是A。',
        )
    ],
    'legal_professional': [
        make_example(
            '未经人民法院依法判决，对任何人都不得确定有罪。这体现的是____原则。',
            '无罪推定',
            '罪刑法定',
            '疑罪从有',
            '自由裁量',
            'A',
            '题干强调“未经依法判决不得确定有罪”，这是无罪推定精神的直接体现。罪刑法定解决的是犯罪与刑罚必须由法律规定，和题干重点不同。因此最终答案是A。',
        )
    ],
    'high_school_chinese': [
        make_example(
            '《师说》的作者是____。',
            '柳宗元',
            '韩愈',
            '欧阳修',
            '苏轼',
            'B',
            '《师说》是唐代韩愈的代表性论说文，这是高中语文中的基础文学常识。柳宗元、欧阳修、苏轼虽然也是重要作家，但不对应这篇作品。因此最终答案是B。',
        )
    ],
    'high_school_history': [
        make_example(
            '《史记》的作者是____。',
            '司马迁',
            '班固',
            '司马光',
            '班昭',
            'A',
            '《史记》由西汉司马迁撰写，是中国第一部纪传体通史。班固主要对应《汉书》，司马光主要对应《资治通鉴》，班昭参与补写《汉书》。所以最终答案是A。',
        )
    ],
    'middle_school_history': [
        make_example(
            '秦统一后推行的标准文字是____。',
            '甲骨文',
            '小篆',
            '楷书',
            '行书',
            'B',
            '秦始皇统一六国后，为加强中央集权，在文字上推行小篆作为规范文字。楷书和行书形成更晚，甲骨文则更早，因此最终答案是B。',
        )
    ],
    'civil_servant': [
        make_example(
            '向上级机关汇报工作、反映情况时通常使用的公文文种是____。',
            '报告',
            '通知',
            '通告',
            '公告',
            'A',
            '题干场景是“向上级汇报工作、反映情况”，对应的标准公文文种就是报告。通知多用于传达事项，通告和公告面向范围也不同。因此最终答案是A。',
        )
    ],
    'sports_science': [
        make_example(
            '运动前进行准备活动的主要作用之一是____。',
            '提高体温和神经兴奋性',
            '立即消除全部疲劳',
            '完全避免任何运动损伤',
            '显著降低心率到静息以下',
            'A',
            '准备活动能让机体逐步进入运动状态，提高体温、加快循环并提升神经肌肉兴奋性。它能降低受伤风险，但不能保证“完全避免”损伤，更不会把心率降到静息以下。因此最终答案是A。',
        )
    ],
    'plant_protection': [
        make_example(
            '蚜虫通常属于____害虫。',
            '刺吸式口器',
            '咀嚼式口器',
            '虹吸式口器',
            '嚼吸式口器',
            'A',
            '蚜虫取食植物汁液，依靠的是刺吸式口器，这是植保中的基础分类知识。咀嚼式更常见于咬食叶片的害虫，不符合蚜虫特点。因此最终答案是A。',
        )
    ],
    'basic_medicine': [
        make_example(
            '在常见体温测量部位中，最接近人体核心体温的是____。',
            '腋下温度',
            '口腔温度',
            '直肠温度',
            '额头皮肤温度',
            'C',
            '核心体温反映体内真实温度，越接近体腔内部的测量部位，受外界环境影响越小。直肠温度通常最接近核心体温，口腔和腋下更易受外界影响，额温受环境影响更明显。因此最终答案是C。',
        )
    ],
    'clinical_medicine': [
        make_example(
            '临床上血压常用的计量单位是____。',
            'mmHg',
            'kg',
            'L',
            'mol',
            'A',
            '血压测量反映的是压力大小，临床习惯单位是毫米汞柱，即mmHg。kg是质量单位，L是体积单位，mol是物质的量单位，都不适合描述血压。因此最终答案是A。',
        )
    ],
    'urban_and_rural_planner': [
        make_example(
            '控制性详细规划的核心作用之一是____。',
            '对土地使用和开发强度提出控制要求',
            '只讨论抽象城市发展愿景',
            '完全替代建筑设计图纸',
            '只负责统计人口而不管空间布局',
            'A',
            '控制性详细规划强调对地块性质、容积率、建筑密度等指标提出控制要求，直接服务于后续开发建设管理。B、C、D都偏离了控规的法定功能，因此最终答案是A。',
        )
    ],
    'accountant': [
        make_example(
            '下列等式中，正确反映基本会计等式的是____。',
            '资产=负债+所有者权益',
            '资产=收入-费用',
            '利润=资产+负债',
            '负债=资产+所有者权益',
            'A',
            '基本会计等式反映企业在某一时点的财务状况，即资产来源于负债和所有者权益。B是把利润关系写错了，C把经营成果和财务状况混在一起，D则把等式方向颠倒了。因此最终答案是A。',
        )
    ],
    'fire_engineer': [
        make_example(
            '在未切断电源前，扑救带电电气设备火灾通常不宜直接使用____。',
            '干粉灭火器',
            '二氧化碳灭火器',
            '水',
            '1211替代灭火介质',
            'C',
            '带电电气火灾首先要考虑导电风险。水具有导电性，未断电前直接用水扑救容易造成触电和扩大事故。干粉、二氧化碳等更常用于此类场景，因此最终答案是C。',
        )
    ],
    'environmental_impact_assessment_engineer': [
        make_example(
            '环境影响评价的核心任务之一是____。',
            '预测并评价建设项目可能造成的环境影响',
            '只在项目建成后统计利润',
            '替代一切工程设计工作',
            '仅负责宣传企业形象',
            'A',
            '环境影响评价的重点就是在项目实施前识别、预测和评价可能产生的环境影响，并提出减缓措施。B、C、D都偏离了环评工作的法定定位，因此最终答案是A。',
        )
    ],
    'tax_accountant': [
        make_example(
            '增值税属于____。',
            '流转税',
            '所得税',
            '财产税',
            '资源税',
            'A',
            '增值税在商品和服务流转环节征收，因此属于流转税。所得税针对所得征收，财产税和资源税对应的征税对象也不同，所以最终答案是A。',
        )
    ],
    'physician': [
        make_example(
            '下列项目中，不属于体格检查基本方法的是____。',
            '视诊',
            '触诊',
            '叩诊',
            '治疗',
            'D',
            '体格检查的基本方法通常包括视诊、触诊、叩诊、听诊等。治疗属于临床处置行为，不是检查方法本身。因此前三项都属于检查手段，最终答案是D。',
        )
    ],
}


missing_subjects = set(ceval_all_sets) - set(manual_fewshot_examples)
assert not missing_subjects, f'Missing few-shot examples for: {sorted(missing_subjects)}'


def build_user_prompt(subject_cn: str, question: str, options: dict) -> str:
    return (
        f'以下是中国关于{subject_cn}考试的单项选择题。'
        '请先输出<think>\n你的思考</think>。'
        '思考过程尽量精炼（建议80字以内），并确保输出闭合的</think>标签。'
        '最后单独一行只输出最终正确选项的字母（A/B/C/D），不要输出解释。\n'
        f'{question}\n'
        f'A. {options["A"]}\n'
        f'B. {options["B"]}\n'
        f'C. {options["C"]}\n'
        f'D. {options["D"]}\n'
        '答案:'
    )


def build_fewshot_rounds(subject_name: str) -> list:
    subject_cn = ceval_subject_mapping[subject_name][1]
    rounds = []
    for example in manual_fewshot_examples[subject_name]:
        rounds.extend([
            dict(
                role='HUMAN',
                prompt=build_user_prompt(subject_cn, example['question'],
                                         example['options']),
            ),
            dict(
                role='BOT',
                prompt=(
                    f'<think>\n{example["think"]}</think>\n\n'
                    f'{example["answer"]}'
                ),
            ),
        ])
    return rounds


def build_auto_think(subject_cn: str) -> str:
    return (
        f'先把这道{subject_cn}题的考点识别出来，再看题干真正问的是概念、'
        '性质、公式、条件还是结论。接着逐项比较A、B、C、D与题干要求是否一致，'
        '排除与定义不符、适用条件不对或只说对一部分的选项。'
        '综合题干和四个选项后，可以确定正确答案是{answer}。'
    )


ceval_datasets = []
for _split in ['val']:
    for _name in ceval_all_sets:
        _ch_name = ceval_subject_mapping[_name][1]
        user_prompt = (
            f'以下是中国关于{_ch_name}考试的单项选择题。'
            '请先输出<think>\n你的思考</think>。'
            '思考过程尽量精炼（建议80字以内），并确保输出闭合的</think>标签。'
            '再单独输出最终正确选项的字母（A/B/C/D）。'
            '不要输出额外解释。\n'
            '{question}\n'
            'A. {A}\n'
            'B. {B}\n'
            'C. {C}\n'
            'D. {D}\n'
            '答案:'
        )
        ceval_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=user_prompt,
                        ),
                        dict(
                            role='BOT',
                            prompt=(
                                f'<think>\n{build_auto_think(_ch_name)}\n'
                                '</think>\n\n{answer}'
                            ),
                        ),
                    ],
                ),
                ice_token='</E>',
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=['</E>'] + build_fewshot_rounds(_name),
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=user_prompt,
                        ),
                        dict(
                            role='BOT',
                            begin='<|im_start|>Assistant: <think>\n',
                            prompt='',
                        ),
                    ],
                ),
                ice_token='</E>',
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3]),
            inferencer=dict(
                type=GenInferencer,
                max_out_len=4096,
                stopping_criteria=[
                    '<|im_end|>',
                    '<|endoftext|>',
                ],
            ),
        )

        ceval_eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(
                type='think-option',
                options='ABCD',
                stop_words=[
                    '<|im_end|>',
                    '<|endoftext|>',
                    '<|im_start|>',
                ],
            ),
        )

        ceval_datasets.append(
            dict(
                type=CEvalDataset,
                path='opencompass/ceval-exam',
                name=_name,
                abbr='ceval-' + _name if _split == 'val' else 'ceval-test-' +
                _name,
                reader_cfg=dict(
                    input_columns=['question', 'A', 'B', 'C', 'D'],
                    output_column='answer',
                    train_split='dev',
                    test_split=_split,
                ),
                infer_cfg=ceval_infer_cfg,
                eval_cfg=ceval_eval_cfg,
            ))

del _split, _name, _ch_name
