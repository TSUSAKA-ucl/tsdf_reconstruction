    def callback(self, msg):
        avg_count = self.get_parameter('avg_frames').value
        z_min = self.get_parameter('z_min').value
        z_max = self.get_parameter('z_max').value

        # PointCloud2 を numpy 配列に変換 (構造を維持するため read_points を利用)
        # 注意: 欠損値(NaN)を含めて読み込むことで行列サイズを固定する
        gen = pc2.read_points(msg, field_names=['x', 'y', 'z', 'rgb'], skip_nans=False)
        data = np.fromiter(gen, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        
        # XYZを抽出
        curr_points = np.stack([data['x'], data['y'], data['z']], axis=1)
        
        # リングバッファの更新
        self.buffer.append(curr_points)
        if len(self.buffer) > avg_count:
            self.buffer.popleft()

        # 指定フレーム数溜まったら平均化して出力
        if len(self.buffer) >= avg_count:
            # 全フレームの平均 (NaNは無視して平均をとる nanmean)
            avg_xyz = np.nanmean(np.array(self.buffer), axis=0)
            
            # Cropping & NaN 除去
            mask = (avg_xyz[:, 2] > z_min) & (avg_xyz[:, 2] < z_max) & (~np.isnan(avg_xyz[:, 2]))
            filtered_xyz = avg_xyz[mask]
            filtered_rgb = data['rgb'][mask] # 色は最新フレームのものを流用（または平均化）

            # PointCloud2 メッセージの再構築
            # 出力データを作成
            out_data = []
            for i in range(len(filtered_xyz)):
                out_data.append([filtered_xyz[i,0], filtered_xyz[i,1], filtered_xyz[i,2], filtered_rgb[i]])
            
            out_msg = pc2.create_cloud(msg.header, msg.fields, out_data)
            self.pub.publish(out_msg)
            
            self.get_logger().info(f'Published averaged cloud ({len(self.buffer)} frames)', once=False)
