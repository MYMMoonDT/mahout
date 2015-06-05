package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;

public class AdjustedPreferenceInferrer implements PreferenceInferrer {

	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {
	}

	@Override
	public float inferPreference(long userID, long itemID)
			throws TasteException {
		return 0;
	}
	
}
