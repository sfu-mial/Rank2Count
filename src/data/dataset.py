from src.helper import get_paired_image_ds, get_density_ds, get_count_image_ds

class CountDataIterator(object):
    def __init__(self, dataset, path, size):
        self.image = [path + d["image"] for d in dataset]
        self.label = [d["count"] for d in dataset]
        self.size = size

    def build(self, batch_size, drop_remainder, shuffle):
        count_ds = get_count_image_ds(self.image, self.label, self.size, shuffle)
        count_ds = count_ds.batch(batch_size, drop_remainder=drop_remainder)
        return count_ds.prefetch(AUTOTUNE)

class RankDataIterator(object):
    def __init__(self, dataset, path, size):
        self.src_f = [path + d["source_image"] for d in dataset]
        self.tar_f = [path + d["target_image"] for d in dataset]
        self.label = [d["label"] for d in dataset]
        self.size = size

    def build(self, batch_size, drop_remainder, shuffle):
        rank_ds = get_paired_image_ds(self.src_f, self.tar_f, self.label, self.size, shuffle)
        rank_ds = rank_ds.batch(batch_size, drop_remainder=drop_remainder)
        return rank_ds.prefetch(AUTOTUNE)
    
class DensityDataIterator(RankDataIterator):
    def __init__(self, dataset, path, size):
        super().__init__(dataset, path, size)
    
    def build(self, batch_size, drop_remainder, shuffle):
        # rank_ds = (src, tar, label)
        rank_ds = get_paired_image_ds(self.src_f, self.tar_f, self.label, self.size, shuffle)
        # dmap_ds = (dmap)
        dmap_ds = get_density_ds(len(self.src_f))
        # rank_dmap_ds = (src, tar, label, dmap)
        rank_dmap_ds = Dataset.zip((rank_ds, dmap_ds)).map(lambda x, y: (x[0], x[1], x[2], y), num_parallel_calls=AUTOTUNE)
        rank_dmap_ds = rank_dmap_ds.batch(batch_size, drop_remainder=drop_remainder)
        return rank_ds.prefetch(AUTOTUNE)