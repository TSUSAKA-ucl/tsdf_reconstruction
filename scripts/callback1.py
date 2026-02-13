    def callback(self, msg):
        avg_count = self.get_parameter('avg_frames').value
        z_min = self.get_parameter('z_min').value
        z_max = self.get_parameter('z_max').value
        
        # 1. PointCloud2 を読み込み (NaNや0を含めて保持)
        gen = pc2.read_points(msg, field_names=['x', 'y', 'z', 'rgb'], skip_nans=False)
        data = np.fromiter(gen, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        curr_xyz = np.stack([data['x'], data['y'], data['z']], axis=1)

        # 2. バッファに貯める
        self.buffer.append(curr_xyz)
        if len(self.buffer) > avg_count:
            self.buffer.popleft()

        if len(self.buffer) >= avg_count:
            # 配列化 (Frames, Points, 3)
            stacked = np.array(self.buffer)
            
            # 3. 0（無効値）を NaN に置換して統計計算から除外する
            # これをしないと、0 と 1.0m の平均が 0.5m になり原点へ引きずられる
            stacked_nan = stacked.copy()
            stacked_nan[stacked_nan == 0] = np.nan
            
            # 4. 平均と標準偏差の計算 (axis=0 は時間方向)
            avg_xyz = np.nanmean(stacked_nan, axis=0)
            std_xyz = np.nanstd(stacked_nan, axis=0)
            
            # 5. フィルタリング条件
            # - Z軸の標準偏差が 0.05 (5cm) 以下（＝ジタバタしていない）
            # - 有効なサンプル数が半分以上ある（＝たまにしか映らない点は除外）
            valid_samples = np.count_nonzero(~np.isnan(stacked_nan[:, :, 2]), axis=0)
            
            std_thresh = 0.02  # 2cm 以上の揺れがある点は不安定とみなす（調整可）
            mask = (std_xyz[:, 2] < std_thresh) & \
                   (valid_samples > (avg_count / 2)) & \
                   (avg_xyz[:, 2] > z_min) & \
                   (avg_xyz[:, 2] < z_max)

            # 6. マスク適用
            filtered_xyz = avg_xyz[mask]
            filtered_rgb = data['rgb'][mask]

            # 7. メッセージ構築 (省略: 前回と同様)
            self.publish_cloud(msg.header, msg.fields, filtered_xyz, filtered_rgb)
