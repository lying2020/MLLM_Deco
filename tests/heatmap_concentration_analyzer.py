import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class HeatmapConcentrationAnalyzer:
    """
    Heatmap集中度分析器
    支持输入：24×24数组、576一维数组、或图像文件路径
    """

    def __init__(self, top_percentile=10, weights=None):
        """
        初始化分析器

        参数:
        - top_percentile: 高像素点的百分比阈值 (默认10%)
        - weights: 各指标的权重 [紧凑度, 大小系数, 连通性系数] (默认[0.5, 0.3, 0.2])
        """
        self.top_percentile = top_percentile

        if weights is None:
            self.weights = [0.5, 0.3, 0.2]  # 紧凑度, 大小系数, 连通性系数
        else:
            self.weights = weights

    def preprocess_heatmap(self, heatmap_input):
        """
        预处理heatmap输入，统一转换为24×24的numpy数组

        支持输入类型:
        1. 24×24的numpy数组
        2. 576个元素的一维数组/列表
        3. 24×24的二维列表
        """
        # 转换为numpy数组
        data = np.array(heatmap_input)

        # 检查并重塑形状
        if data.size == 576 and data.shape != (24, 24):
            # 如果是576个元素的一维数组
            if data.ndim == 1:
                data = data.reshape(24, 24)
            elif data.ndim == 2 and (data.shape[0] == 24 or data.shape[1] == 24):
                # 可能是其他形状的24×24变形
                data = data.reshape(24, 24)
            else:
                raise ValueError(f"输入形状 {data.shape} 无法转换为24×24")

        elif data.shape != (24, 24):
            raise ValueError(f"期望24×24的形状，但得到 {data.shape}")

        return data

    def visualize_heatmap(self, heatmap, binary_map=None, title="Heatmap"):
        """
        可视化heatmap和高像素点区域
        """
        fig, axes = plt.subplots(1, 3 if binary_map is not None else 1,
                                figsize=(15, 5 if binary_map is not None else 5))

        if binary_map is None:
            ax = axes if isinstance(axes, plt.Axes) else axes[0]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        else:
            # 显示原始heatmap
            im1 = axes[0].imshow(heatmap, cmap='hot', interpolation='nearest')
            axes[0].set_title(f'Original {title}')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])

            # 显示二值化图
            axes[1].imshow(binary_map, cmap='gray', interpolation='nearest')
            axes[1].set_title(f'Top {self.top_percentile}% Pixels')
            axes[1].axis('off')

            # 显示叠加图
            overlay = np.zeros((*heatmap.shape, 3))
            overlay[:,:,0] = heatmap / heatmap.max()  # 红色通道: heatmap
            overlay[:,:,1] = binary_map  # 绿色通道: 高像素点区域
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay: Heatmap + High Pixels')
            axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def extract_high_pixels(self, heatmap):
        """
        提取高像素点区域
        """
        # 计算阈值
        threshold = np.percentile(heatmap, 100 - self.top_percentile)

        # 创建二值化掩码
        binary_map = (heatmap >= threshold).astype(np.uint8)

        return binary_map, threshold

    def calculate_concentration_index(self, heatmap_input, visualize=False, top_k=6):
        """
        计算heatmap的像素集中度指数（基于方案C：卷积+聚类中心+权重筛选）

        参数:
        - heatmap_input: 输入heatmap (多种格式)
        - visualize: 是否可视化结果
        - top_k: 选取的Top-K个参考点数量（默认4）

        返回:
        - dict: 包含各项指标和综合集中度
        """
        # 1. 预处理输入
        heatmap = self.preprocess_heatmap(heatmap_input)

        # 检查heatmap是否全为0或无效
        if np.all(heatmap == 0) or np.all(np.isnan(heatmap)):
            return {
                'concentration_index': 0.0,
                'spatial_compactness': 0.0,
                'pixel_weight_ratio': 0.0,
                'message': 'Empty or invalid heatmap'
            }

        # 1.5. 将像素值最高的10个点置为0，避免异常过大的值的干扰
        # 找到最高的10个点的索引（排除NaN值）
        flat_heatmap = heatmap.flatten()
        # 创建一个有效值的掩码（非NaN且大于0）
        valid_mask = ~np.isnan(flat_heatmap) & (flat_heatmap > 0)
        if np.sum(valid_mask) > 0:
            # 只对有效值进行排序，找到最高的10个
            valid_values = flat_heatmap[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            # 获取最高的10个值的索引（从大到小）
            num_top = min(10, len(valid_values))
            top_indices_in_valid = np.argsort(valid_values)[-num_top:][::-1]
            top_10_flat_indices = valid_indices[top_indices_in_valid]
            # 转换为2D坐标
            top_10_positions = np.unravel_index(top_10_flat_indices, heatmap.shape)
            # 将这10个点的值置为0
            for i, j in zip(top_10_positions[0], top_10_positions[1]):
                heatmap[i, j] = 0.0

        # 2. 4×4卷积（全1核，stride=1，padding=0）→ 20×20
        kernel = np.ones((4, 4))
        convolved = convolve2d(heatmap, kernel, mode='valid')  # 得到20×20

        # 检查卷积结果是否有效
        if convolved.size == 0 or np.all(np.isnan(convolved)):
            return {
                'concentration_index': 0.0,
                'spatial_compactness': 0.0,
                'pixel_weight_ratio': 0.0,
                'message': 'Convolution failed'
            }

        # 3. 选取Top-K个点（从20×20中）
        flat_indices = np.argsort(convolved.flatten())[-top_k:]
        top_k_flat_indices = flat_indices[::-1]  # 从高到低排序
        top_k_positions_20x20 = np.unravel_index(top_k_flat_indices, convolved.shape)  # (row_indices, col_indices)

        # 4. 映射回24×24坐标
        # 卷积后的坐标(i, j)对应原始heatmap的(i, j)到(i+3, j+3)的4×4区域
        # 我们使用区域中心点作为映射坐标：(i+2, j+2)
        top_k_positions_24x24 = []
        for i, j in zip(top_k_positions_20x20[0], top_k_positions_20x20[1]):
            # 映射到原始坐标（确保在边界内）
            orig_i = min(i + 2, heatmap.shape[0] - 1)
            orig_j = min(j + 2, heatmap.shape[1] - 1)
            top_k_positions_24x24.append((orig_i, orig_j))

        # 5. 收集候选点集合（K个参考点+每个点的8邻域，不去重）
        candidate_points = []
        for ref_i, ref_j in top_k_positions_24x24:
            # 添加参考点本身
            candidate_points.append((ref_i, ref_j))
            # 添加8邻域
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # 跳过参考点本身
                    ni, nj = ref_i + di, ref_j + dj
                    # 检查边界
                    if 0 <= ni < heatmap.shape[0] and 0 <= nj < heatmap.shape[1]:
                        candidate_points.append((ni, nj))

        if len(candidate_points) == 0:
            return {
                'concentration_index': 0.0,
                'spatial_compactness': 0.0,
                'pixel_weight_ratio': 0.0,
                'message': 'No candidate points found'
            }

        # 6. 计算所有候选点到K个参考点的距离方差，选方差最小的作为聚类中心
        def euclidean_distance(p1, p2):
            """计算两点之间的欧氏距离"""
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        min_variance = float('inf')
        cluster_center = None

        for ref_point in top_k_positions_24x24:
            # 计算所有候选点到该参考点的距离
            distances = [euclidean_distance(cand_point, ref_point) for cand_point in candidate_points]
            distance_variance = np.var(distances)

            if distance_variance < min_variance:
                min_variance = distance_variance
                cluster_center = ref_point

        if cluster_center is None:
            cluster_center = top_k_positions_24x24[0]  # 默认使用第一个参考点

        # 7. 计算候选点到聚类中心的权重距离（logit*距离），剔除最远的10%
        weighted_distances = []
        for cand_point in candidate_points:
            distance = euclidean_distance(cand_point, cluster_center)
            logit_value = 1.0 # = heatmap[cand_point[0], cand_point[1]]
            # 处理NaN和负值
            if np.isnan(logit_value) or logit_value < 0:
                logit_value = 0.0
            weighted_distance = logit_value * distance
            weighted_distances.append((cand_point, weighted_distance))

        # 按权重距离排序
        weighted_distances.sort(key=lambda x: x[1], reverse=True)

        # 剔除最远的10%
        remove_count = max(1, int(len(weighted_distances) * 0.1))
        filtered_candidate_points = [point for point, _ in weighted_distances[:-remove_count]]

        if len(filtered_candidate_points) == 0:
            return {
                'concentration_index': 0.0,
                'spatial_compactness': 0.0,
                'pixel_weight_ratio': 0.0,
                'message': 'No points after filtering'
            }

        # 8. 计算两个指标
        # 8.1 空间指标：候选点的空间分布紧凑度
        if len(filtered_candidate_points) == 1:
            spatial_compactness = 1.0  # 单点，完全紧凑
        else:
            # 计算候选点的边界框
            y_coords = [p[0] for p in filtered_candidate_points]
            x_coords = [p[1] for p in filtered_candidate_points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            bounding_box_width = max_x - min_x + 1
            bounding_box_height = max_y - min_y + 1
            bounding_box_area = bounding_box_width * bounding_box_height

            # 计算到质心的平均距离
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            distances_to_centroid = [euclidean_distance(p, (centroid_y, centroid_x))
                                    for p in filtered_candidate_points]
            mean_distance = np.mean(distances_to_centroid)

            # 计算距离的标准差（衡量分散程度）
            distance_std = np.std(distances_to_centroid) if len(distances_to_centroid) > 1 else 0.0

            # 紧凑度计算：
            # 1. 密度：点数/边界框面积（点数越多、边界框越小，密度越高）
            if bounding_box_area > 0:
                density = len(filtered_candidate_points) / bounding_box_area
            else:
                density = 1.0

            # 2. 距离因子：平均距离越小，紧凑度越高
            # 使用归一化：假设最大可能距离为对角线长度
            max_possible_distance = np.sqrt(heatmap.shape[0]**2 + heatmap.shape[1]**2)
            normalized_mean_distance = mean_distance / max_possible_distance if max_possible_distance > 0 else 0.0
            distance_factor = 1.0 / (1.0 + normalized_mean_distance)

            # 3. 分散因子：标准差越小，越集中
            normalized_std = distance_std / max_possible_distance if max_possible_distance > 0 else 0.0
            dispersion_factor = 1.0 / (1.0 + normalized_std)

            # 综合紧凑度：密度 × 距离因子 × 分散因子
            # 归一化：理论上限需要根据实际情况调整，这里先简单归一化
            spatial_compactness = density * distance_factor * dispersion_factor
            # 归一化到[0, 1]（理论上限约为1.0，但实际可能超过，所以需要clamp）
            spatial_compactness = min(1.0, spatial_compactness)

        # 8.2 像素权重指标：候选点的总像素值占整张图的像素值比例
        candidate_pixel_sum = sum(heatmap[p[0], p[1]] for p in filtered_candidate_points
                                  if not np.isnan(heatmap[p[0], p[1]]))
        total_pixel_sum = np.nansum(heatmap)

        if total_pixel_sum > 0:
            pixel_weight_ratio = candidate_pixel_sum / total_pixel_sum
        else:
            pixel_weight_ratio = 0.0

        print(f"spatial_compactness: {spatial_compactness}, pixel_weight_ratio: {pixel_weight_ratio}")
        # 9. 计算综合集中度指数（两个指标相加）
        # 由于两个指标都在[0, 1]范围内，可以直接相加，然后归一化
        concentration_index = spatial_compactness + pixel_weight_ratio
        # 归一化到[0, 1]（因为两个指标相加最大为2）
        concentration_index = concentration_index / 2.0

        # 10. 收集结果
        result = {
            'concentration_index': float(concentration_index),
            'spatial_compactness': float(spatial_compactness),
            'pixel_weight_ratio': float(pixel_weight_ratio),
            'cluster_center': cluster_center,
            'filtered_candidate_count': len(filtered_candidate_points),
            'total_candidate_count': len(candidate_points),
            'top_k_positions': top_k_positions_24x24,
            'heatmap_shape': heatmap.shape
        }

        # 11. 可视化（如果启用）
        if visualize:
            # 创建二值化图显示候选点
            binary_map = np.zeros_like(heatmap)
            for point in filtered_candidate_points:
                binary_map[point[0], point[1]] = 1
            title = f"Concentration Index: {concentration_index:.3f}"
            self.visualize_heatmap(heatmap, binary_map, title)

        return result

    def calculate_concentration_index_enhanced(self, heatmap_input, visualize=False, use_gaussian=True, use_classical_metrics=True):
        """
        增强版集中度计算：结合当前方法和经典统计学指标

        方法对比：
        ==========
        当前方法 (calculate_concentration_index):
        - 优点：基于空间聚类和局部邻域，能捕捉空间聚集模式
        - 缺点：计算复杂，缺乏统计学理论基础，对参数敏感

        经典方法 (Gini + Moran's I + Peak Concentration):
        - 优点：统计学基础扎实，可解释性强，计算稳定
        - 缺点：可能忽略局部空间结构

        本方法结合两者优点：
        1. 使用高斯平滑预处理（替代全1核卷积，更符合信号处理理论）
        2. 计算基尼系数（衡量值分布的不平等性）
        3. 计算空间紧凑度（基于连通区域分析）
        4. 计算峰值聚集度（高注意力区域的连通性）
        5. 综合指标 = w1*Gini + w2*Spatial_Compactness + w3*Peak_Concentration

        参数:
        - heatmap_input: 输入heatmap (多种格式)
        - visualize: 是否可视化结果
        - use_gaussian: 是否使用高斯平滑预处理（默认True）
        - use_classical_metrics: 是否使用经典统计学指标（默认True）

        返回:
        - dict: 包含各项指标和综合集中度
        """
        # 1. 预处理输入
        heatmap = self.preprocess_heatmap(heatmap_input)

        # 检查heatmap是否全为0或无效
        if np.all(heatmap == 0) or np.all(np.isnan(heatmap)):
            return {
                'concentration_index': 0.0,
                'gini_coefficient': 0.0,
                'spatial_compactness': 0.0,
                'peak_concentration': 0.0,
                'high_attention_ratio': 0.0,
                'message': 'Empty or invalid heatmap'
            }

        # 1.5. 将像素值最高的10个点置为0，避免异常过大的值的干扰
        flat_heatmap = heatmap.flatten()
        valid_mask = ~np.isnan(flat_heatmap) & (flat_heatmap > 0)
        if np.sum(valid_mask) > 0:
            valid_values = flat_heatmap[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            num_top = min(10, len(valid_values))
            top_indices_in_valid = np.argsort(valid_values)[-num_top:][::-1]
            top_10_flat_indices = valid_indices[top_indices_in_valid]
            top_10_positions = np.unravel_index(top_10_flat_indices, heatmap.shape)
            for i, j in zip(top_10_positions[0], top_10_positions[1]):
                heatmap[i, j] = 0.0

        # 2. 高斯平滑预处理（替代全1核卷积）
        if use_gaussian:
            # 使用高斯核平滑，sigma=1.5（经验值，可根据需要调整）
            smoothed_heatmap = gaussian_filter(heatmap, sigma=1.5)
        else:
            smoothed_heatmap = heatmap.copy()

        # 3. 计算基尼系数（Gini Coefficient）- 衡量值分布的不平等性
        # 基尼系数范围[0, 1]，0表示完全平等，1表示完全不平等
        # 对于注意力热力图，值越大表示注意力越集中
        flattened = smoothed_heatmap.flatten()
        valid_values = flattened[~np.isnan(flattened) & (flattened >= 0)]

        if len(valid_values) == 0 or np.sum(valid_values) == 0:
            gini = 0.0
        else:
            # 排序并计算基尼系数
            sorted_values = np.sort(valid_values)
            n = len(sorted_values)
            cum_values = np.cumsum(sorted_values)
            if cum_values[-1] > 0:
                # 基尼系数公式：G = (n+1-2*Σ(i*yi)/Σyi) / n
                gini = (n + 1 - 2 * np.sum((np.arange(1, n+1) * sorted_values) / cum_values[-1])) / n
                gini = max(0.0, min(1.0, gini))  # 确保在[0, 1]范围内
            else:
                gini = 0.0

        # 4. 计算空间紧凑度（基于连通区域分析）
        # 使用阈值分割，找到高注意力区域
        threshold = np.nanpercentile(smoothed_heatmap, 90)  # 前10%作为高注意力区域
        high_attention_mask = (smoothed_heatmap > threshold) & (~np.isnan(smoothed_heatmap))
        high_attention_ratio = np.mean(high_attention_mask)

        if np.any(high_attention_mask):
            # 标记连通区域
            labeled_array, num_features = label(high_attention_mask)
            if num_features > 0:
                # 计算最大连通区域的面积
                region_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
                largest_region_size = max(region_sizes) if region_sizes else 0
                total_high_attention_pixels = np.sum(high_attention_mask)

                # 峰值聚集度：最大连通区域占高注意力区域的比例
                # 值越大，表示高注意力区域越集中在一个连通块中
                peak_concentration = largest_region_size / total_high_attention_pixels if total_high_attention_pixels > 0 else 0.0

                # 空间紧凑度：考虑连通区域的数量和分布
                # 连通区域越少，空间越紧凑
                num_regions_factor = 1.0 / (1.0 + num_features)  # 区域越少，因子越大
                # 最大区域占比
                largest_ratio = largest_region_size / total_high_attention_pixels if total_high_attention_pixels > 0 else 0.0
                spatial_compactness = num_regions_factor * largest_ratio
            else:
                peak_concentration = 0.0
                spatial_compactness = 0.0
        else:
            peak_concentration = 0.0
            spatial_compactness = 0.0

        # 5. 计算综合集中度指数
        # 权重可以根据实际需求调整
        w1, w2, w3 = 0.4, 0.3, 0.3  # Gini, Spatial_Compactness, Peak_Concentration
        concentration_index = (w1 * gini +
                             w2 * spatial_compactness +
                             w3 * peak_concentration)

        # 归一化到[0, 1]
        concentration_index = max(0.0, min(1.0, concentration_index))

        # 6. 收集结果
        result = {
            'concentration_index': float(concentration_index),
            'gini_coefficient': float(gini),
            'spatial_compactness': float(spatial_compactness),
            'peak_concentration': float(peak_concentration),
            'high_attention_ratio': float(high_attention_ratio),
            'threshold_used': float(threshold),
            'heatmap_shape': heatmap.shape,
            'method': 'enhanced_classical_metrics'
        }

        # 7. 可视化（如果启用）
        if visualize:
            binary_map = np.zeros_like(smoothed_heatmap)
            binary_map[high_attention_mask] = 1
            title = f"Enhanced CI: {concentration_index:.3f} (Gini: {gini:.3f}, Compact: {spatial_compactness:.3f}, Peak: {peak_concentration:.3f})"
            self.visualize_heatmap(smoothed_heatmap, binary_map, title)

        return result
