pub fn decomp_split_comps(&mut self, g: &G, g_comps: &Vec<G>, depth: usize) -> &mut Self {
    let mut nterms_comps = 0;
    let mut scalar_comps = g.scalar().clone();
    for h in g_comps {
        let mut d = Decomposer::new(h);
        d.use_cats(self.use_cats);
        d.split_comps(self.split_comps);
        d.use_heur(self.use_heur);
        d.with_full_simp();

        let depth = if depth > self.max_depth {
            depth - self.max_depth
        } else {
            0
        };

        let d = d.decomp_parallel(depth);

        nterms_comps += d.nterms;
        scalar_comps *= d.scalar;
    }
    self.nterms += nterms_comps;
    self.scalar = &self.scalar + scalar_comps;

    self
}