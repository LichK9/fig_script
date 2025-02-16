import glob
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
# from calculate_reuse_distance import calculate_reuse_distance

def get_input_files(input_pattern) -> List[str]:
    """获取所有输入文件"""
    files = glob.glob(input_pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")
    return sorted(files)

def process_traces(trace_files):
    """
    并行处理多个trace文件
    """
    # trace_files = [os.path.join(trace_directory, f) for f in os.listdir(trace_directory) 
    #               if f.endswith('.trace')]
    
    results = []
    paths = ['degreeCentr','graphColoring','kCore','pageRank','triangleCount']
    # max_insts = [45100000,140500000,7700000,35200000,23800000]
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(calculate_reuse_distance, glob.glob(path+'/*/*_llc_trace.csv')[0],[0,16, 32, 64, 128,256,np.inf]) 
                  for path in paths]
        
        for future in futures:
            results.append(future.result())
            
    return results

def plot_reuse_distance(results):
    """
    绘制重用距离统计图
    """
    custom_order = ["605.mcf_s","619.lbm_s","620.omnetpp_s","625.x264_s","638.imagick_s","degreeCentr","graphColoring","kCore","pageRank","triangleCount","433.milc","444.namd","462.libquantum","473.astar"]
    results = sorted(results, key=lambda x: custom_order.index(x['benchmark']))
    xlabels=[r['benchmark'] for r in results]
    ylabels='Reuse Distance Ratio(%)'

    n = len(xlabels)    # number of test systems
    x = np.arange(n) 

    # Get all distance keys
    distance_keys = sorted(results[0]['distances'].keys())
    
    # Initialize result list
    b1 = []
    
    # For each distance key, collect values from all benches
    for key in distance_keys:
        values = [item['distances'][key] for item in results]
        b1.append(values)
        
    print(b1)
        
    width=0.3
    stack_labels = ['0-15', '16-31',"32-63",'64-127','128-255','256+']
    colors = ['#00B050', '#F79646', '#C00000', '#FF0000', '#B9D4E7', '#FFC000']

    allfigsize=(8,2.5)  
    plt.figure(figsize=allfigsize)
    bottom = np.zeros(n)
    # m = len(stack_labels)
    m = len(b1)
    for i in range(m):
        plt.bar(x, b1[i], width, bottom=bottom,label=stack_labels[i], color=colors[i], edgecolor='black', linewidth=1)    
        bottom += b1[i]

    plt.grid(axis='y',linestyle='--')
    plt.xticks(x,xlabels, fontsize=10, rotation=30,ha='right', va='top')  
    # plt.yticks([10,50,100,150],fontsize=13)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1f}%")
    )
    plt.ylabel(ylabels,fontsize=13)    
    plt.legend(loc='upper center', ncol=6,  columnspacing=1.5,bbox_to_anchor=(0.5, 1.17), fontsize=10)
    plt.savefig('PageReuseDistance.pdf',bbox_inches = 'tight')     

def main():
    parser = argparse.ArgumentParser(description='Parallel Python Script Runner')
    parser.add_argument('input_pattern', help='Input file pattern (e.g., "data/*.txt")')

    args = parser.parse_args()

    input_files = get_input_files(args.input_pattern)
    print(f"input_files:{input_files}")

    # 处理所有trace文件
    results = process_traces(input_files)

    print("results",results)

    prev_resutlts = [{'benchmark': '473.astar', 'distances': {0: np.float64(7.422347516722544), 16: np.float64(1.9702384142065326), 32: np.float64(3.0526328542371934), 64: np.float64(7.193495972843193), 128: np.float64(17.721788118198276), 256: np.float64(62.63949712379227)}}, {'benchmark': '638.imagick_s', 'distances': {0: np.float64(82.6283833402948), 16: np.float64(7.228158350864536), 32: np.float64(0.7592179783780983), 64: np.float64(0.7704387513557844), 128: np.float64(1.225792386930517), 256: np.float64(7.388009192176269)}}, {'benchmark': '619.lbm_s', 'distances': {0: np.float64(27.68297132691429), 16: np.float64(32.55112391386006), 32: np.float64(35.445189760923625), 64: np.float64(0.29470051697606736), 128: np.float64(0.9900031841324493), 256: np.float64(3.0360112971935127)}}, {'benchmark': '462.libquantum', 'distances': {0: np.float64(81.11707901460944), 16: np.float64(0.44110639984067745), 32: np.float64(0.37130025510576775), 64: np.float64(0.24248504236426888), 128: np.float64(0.1313270549239924), 256: np.float64(17.696702233155857)}}, {'benchmark': '605.mcf_s', 'distances': {0: np.float64(28.314737143640595), 16: np.float64(5.571929621365891), 32: np.float64(6.128184096405581), 64: np.float64(6.684701615659239), 128: np.float64(7.9551648584517745), 256: np.float64(45.34528266447692)}}, {'benchmark': '433.milc', 'distances': {0: np.float64(95.73933659716603), 16: np.float64(0.03101096949864509), 32: np.float64(0.25607092503075823), 64: np.float64(0.02320619529596596), 128: np.float64(0.15025305307900572), 256: np.float64(3.800122259929605)}}, {'benchmark': '444.namd', 'distances': {0: np.float64(77.9508809940713), 16: np.float64(4.500296301460592), 32: np.float64(3.6744062574578136), 64: np.float64(3.2290155551563156), 128: np.float64(2.7790663688457116), 256: np.float64(7.866334523008279)}}, {'benchmark': '620.omnetpp_s', 'distances': {0: np.float64(59.21169962481684), 16: np.float64(6.756969708801516), 32: np.float64(6.473647641871357), 64: np.float64(5.66505790949958), 128: np.float64(4.899515996443253), 256: np.float64(16.99310911856746)}}, {'benchmark': '625.x264_s', 'distances': {0: np.float64(69.066532626315), 16: np.float64(15.625464678253081), 32: np.float64(8.307872502063779), 64: np.float64(2.603438154096489), 128: np.float64(1.433176226977115), 256: np.float64(2.963515812294546)}}]

    results.extend(prev_resutlts)
    
    # 绘制统计图
    plot_reuse_distance(results)
    
    # 输出一些基本统计信息
    for result in results:
        distances = result['distances']
        # print(f"\nBenchmark: {result['benchmark']}")
        # print(f"Average reuse distance: {np.mean(distances):.2f}")
        # print(f"Median reuse distance: {np.median(distances):.2f}")

if __name__ == "__main__":
    # main()
    # results = [{'benchmark': 'degreeCentr', 'distances': {0: np.float64(41.41621724359127), 16: np.float64(6.758570715829987), 32: np.float64(6.405763702454206), 64: np.float64(6.64156232923903), 128: np.float64(5.785678648642228), 256: np.float64(32.99220736024328)}}, {'benchmark': 'graphColoring', 'distances': {0: np.float64(37.29847468099975), 16: np.float64(3.238087464574266), 32: np.float64(3.0939448640218448), 64: np.float64(3.962769138890463), 128: np.float64(6.730945654608202), 256: np.float64(45.67577819690547)}}, {'benchmark': 'kCore', 'distances': {0: np.float64(42.067883829127304), 16: np.float64(6.4764556605103625), 32: np.float64(6.3654307063301845), 64: np.float64(6.791907514450866), 128: np.float64(6.268504159030029), 256: np.float64(32.02981813055125)}}, {'benchmark': 'pageRank', 'distances': {0: np.float64(47.6114679308881), 16: np.float64(3.7158735937267213), 32: np.float64(3.4019387701914185), 64: np.float64(3.523250425919549), 128: np.float64(5.31275895313469), 256: np.float64(36.43471032613952)}}, {'benchmark': 'triangleCount', 'distances': {0: np.float64(56.082513206930095), 16: np.float64(7.996708765317134), 32: np.float64(4.4708034260509955), 64: np.float64(4.016815782239838), 128: np.float64(9.818689817779697), 256: np.float64(17.614469001682245)}}]
    results = [{'benchmark': 'degreeCentr', 'distances': {0: np.float64(41.41621724359127), 16: np.float64(6.758570715829987), 32: np.float64(6.405763702454206), 64: np.float64(6.64156232923903), 128: np.float64(5.785678648642228), 256: np.float64(32.99220736024328)}}, 
                {'benchmark': 'graphColoring', 'distances': {0: np.float64(37.29847468099975), 16: np.float64(3.238087464574266), 32: np.float64(3.0939448640218448), 64: np.float64(3.962769138890463), 128: np.float64(6.730945654608202), 256: np.float64(45.67577819690547)}}, 
                {'benchmark': 'kCore', 'distances': {0: np.float64(42.067883829127304), 16: np.float64(6.4764556605103625), 32: np.float64(6.3654307063301845), 64: np.float64(6.791907514450866), 128: np.float64(6.268504159030029), 256: np.float64(32.02981813055125)}}, 
                {'benchmark': 'pageRank', 'distances': {0: np.float64(47.6114679308881), 16: np.float64(3.7158735937267213), 32: np.float64(3.4019387701914185), 64: np.float64(3.523250425919549), 128: np.float64(5.31275895313469), 256: np.float64(36.43471032613952)}},
                {'benchmark': 'triangleCount', 'distances': {0: np.float64(56.082513206930095), 16: np.float64(7.996708765317134), 32: np.float64(4.4708034260509955), 64: np.float64(4.016815782239838), 128: np.float64(9.818689817779697), 256: np.float64(17.614469001682245)}},
                {'benchmark': '473.astar', 'distances': {0: np.float64(7.422347516722544), 16: np.float64(1.9702384142065326), 32: np.float64(3.0526328542371934), 64: np.float64(7.193495972843193), 128: np.float64(17.721788118198276), 256: np.float64(62.63949712379227)}}, 
                {'benchmark': '638.imagick_s', 'distances': {0: np.float64(82.6283833402948), 16: np.float64(7.228158350864536), 32: np.float64(0.7592179783780983), 64: np.float64(0.7704387513557844), 128: np.float64(1.225792386930517), 256: np.float64(7.388009192176269)}}, 
                {'benchmark': '619.lbm_s', 'distances': {0: np.float64(27.68297132691429), 16: np.float64(32.55112391386006), 32: np.float64(35.445189760923625), 64: np.float64(0.29470051697606736), 128: np.float64(0.9900031841324493), 256: np.float64(3.0360112971935127)}}, 
                {'benchmark': '462.libquantum', 'distances': {0: np.float64(81.11707901460944), 16: np.float64(0.44110639984067745), 32: np.float64(0.37130025510576775), 64: np.float64(0.24248504236426888), 128: np.float64(0.1313270549239924), 256: np.float64(17.696702233155857)}}, 
                {'benchmark': '605.mcf_s', 'distances': {0: np.float64(28.314737143640595), 16: np.float64(5.571929621365891), 32: np.float64(6.128184096405581), 64: np.float64(6.684701615659239), 128: np.float64(7.9551648584517745), 256: np.float64(45.34528266447692)}}, 
                {'benchmark': '433.milc', 'distances': {0: np.float64(95.73933659716603), 16: np.float64(0.03101096949864509), 32: np.float64(0.25607092503075823), 64: np.float64(0.02320619529596596), 128: np.float64(0.15025305307900572), 256: np.float64(3.800122259929605)}}, 
                {'benchmark': '444.namd', 'distances': {0: np.float64(77.9508809940713), 16: np.float64(4.500296301460592), 32: np.float64(3.6744062574578136), 64: np.float64(3.2290155551563156), 128: np.float64(2.7790663688457116), 256: np.float64(7.866334523008279)}}, 
                {'benchmark': '620.omnetpp_s', 'distances': {0: np.float64(59.21169962481684), 16: np.float64(6.756969708801516), 32: np.float64(6.473647641871357), 64: np.float64(5.66505790949958), 128: np.float64(4.899515996443253), 256: np.float64(16.99310911856746)}}, 
                {'benchmark': '625.x264_s', 'distances': {0: np.float64(69.066532626315), 16: np.float64(15.625464678253081), 32: np.float64(8.307872502063779), 64: np.float64(2.603438154096489), 128: np.float64(1.433176226977115), 256: np.float64(2.963515812294546)}}]
    plot_reuse_distance(results)