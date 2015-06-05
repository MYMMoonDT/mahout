package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;


public final class JMSDSimilarity extends ImprovedAbstractSimilarity {

	public JMSDSimilarity(DataModel dataModel) throws TasteException {
		this(dataModel, Weighting.UNWEIGHTED);
	}
	
	public JMSDSimilarity(DataModel dataModel, Weighting weighting)
			throws TasteException {
		super(dataModel, weighting, false);
	}

	@Override
	double computeResult(int n, int nX, int nY, double sumXY, double sumX2,
			double sumY2, double sumXYdiff2) {
		return ((double)n / (double)(nX + nY - n)) * (1 - (sumXYdiff2 / n));
	}

}
