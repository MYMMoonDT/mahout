package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;

public final class MSDSimilarity extends AbstractSimilarity {

	public MSDSimilarity(DataModel dataModel) throws TasteException {
		this(dataModel, Weighting.UNWEIGHTED);
	}
	
	public MSDSimilarity(DataModel dataModel, Weighting weighting)
			throws TasteException {
		super(dataModel, weighting, false);
	}

	@Override
	double computeResult(int n, double sumXY, double sumX2, double sumY2,
			double sumXYdiff2) {
		return 1 - (sumXYdiff2 / n);
	}

}
