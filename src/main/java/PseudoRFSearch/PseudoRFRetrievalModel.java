package PseudoRFSearch;

import Classes.Document;
import Classes.Query;
import IndexingLucene.MyIndexReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class PseudoRFRetrievalModel {

    private final MyIndexReader ixreader;
    private final long allContentLength;
    private final SearchLucene.QueryRetrievalModel qrm;

    private final double mu = 2000;

    public PseudoRFRetrievalModel(MyIndexReader ixreader) {
        this.ixreader = ixreader;
        this.allContentLength = ixreader.getTotalContentLength();
        this.qrm = new SearchLucene.QueryRetrievalModel(ixreader);
    }

    /**
     * Search for the topic with pseudo relevance feedback in 2017 spring assignment 4.
     * The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
     *
     * @param aQuery The query to be searched for.
     * @param TopN   The maximum number of returned document
     * @param TopK   The count of feedback documents
     * @param alpha  parameter of relevance feedback model
     * @return TopN most relevant document, in List structure
     */
    public List<Document> RetrieveQuery(Query aQuery, int TopN, int TopK, double alpha) throws Exception {
        // this method will return the retrieval result of the given Query, and this result is enhanced with pseudo relevance feedback
        // (1) you should first use the original retrieval model to get TopK documents, which will be regarded as feedback documents
        // (2) implement GetTokenRFScore to get each query token's P(token|feedback model) in feedback documents
        // (3) implement the relevance feedback model for each token: combine the each query token's original retrieval score P(token|document) with its score in feedback documents P(token|feedback model)
        // (4) for each document, use the query likelihood language model to get the whole query's new score, P(Q|document)=P(token_1|document')*P(token_2|document')*...*P(token_n|document')

        // convert query to terms
        String[] tokens = aQuery.GetQueryContent().split(" ");

        // Get posting like in hw3
        HashMap<Integer, HashMap<String, Integer>> postingData = postingMapping(tokens);

        //get P(token|feedback documents)
        HashMap<String, Double> TokenRFScore = GetTokenRFScore(aQuery, TopK, postingData);


        // sort all retrieved documents from most relevant to least, and return TopN
        List<Document> results = new ArrayList<>(TopN);
        results.sort((d1, d2) -> d1.score() > d2.score() ? -1 : 1);
        return results;
    }

    /**
     * for each token in the query, you should calculate token's score in feedback documents: P(token|feedback documents)
     * use Dirichlet smoothing
     * save <token, score> in HashMap TokenRFScore, and return it
     *
     * @param aQuery
     * @param TopK
     * @return
     */
    private HashMap<String, Double> GetTokenRFScore(Query aQuery, int TopK, HashMap<Integer, HashMap<String, Integer>> p) throws Exception {
        HashMap<String, Double> TokenRFScore = new HashMap<>();

        List<Document> originalRes = this.qrm.retrieveQuery(aQuery, TopK);
        HashMap<String, Long> pDoc = new HashMap<>();
        long doclen = populatePseudoDoc(pDoc, originalRes, p);

        // Smooth (come from hw3)
        double adjLen = (doclen + this.mu);
        double // (|D|/(|D|+MU)) as l1 and (MU/(|D|+MU)) as r1
                l1 = 1.0 * doclen / adjLen,
                r1 = 1.0 * this.mu / adjLen;
        for (String token : pDoc.keySet()) {
            long tf = pDoc.get(token);
            long cf = this.ixreader.CollectionFreq(token);
            double // p(w|D) = l1*(c(w,D)/|D|) + r1*p(w|REF)
                    l2 = 1.0 * tf / doclen,
                    r2 = 1.0 * cf / this.allContentLength;
            // Unigram LM
            TokenRFScore.put(token, l1 * l2 + r1 * r2);
        }

        return TokenRFScore;
    }

    private long populatePseudoDoc(HashMap<String, Long> doc, List<Document> orig, HashMap<Integer, HashMap<String, Integer>> postings) throws IOException {
        long totalLen = 0;
        for (Document d : orig) {
            int docId = Integer.valueOf(d.docid());
            HashMap<String, Integer> posting = postings.get(docId);
            if (posting == null) {
                System.err.println("Document not in posting list");
                continue;
            }
            for (String token : posting.keySet()) {
                int tf = posting.getOrDefault(token, 0);
                long old = doc.getOrDefault(token, 0L);
                doc.put(token, old + tf);
            }
            totalLen += this.ixreader.docLength(docId);
        }
        return totalLen;
    }

    /**
     * Code from hw3
     *
     * @return <DOCID, <TERM, FREQ>>
     */
    private HashMap<Integer, HashMap<String, Integer>> postingMapping(String[] tokens) throws IOException {
        HashMap<Integer, HashMap<String, Integer>> tokenOnCollection = new HashMap<>();
        for (String token : tokens) {
            Long cFreq = this.ixreader.CollectionFreq(token);
            // Non-exist, no need to calc posting list
            if (cFreq.equals(0L)) {
                System.err.println(String.format("Term <%s> not appeared.", token));
                continue;
            }
            int[][] postingList = this.ixreader.getPostingList(token);
            for (int[] postingForOneDoc : postingList) {
                int docid = postingForOneDoc[0], docFreq = postingForOneDoc[1];
                HashMap<String, Integer> oneTermFreq = tokenOnCollection.getOrDefault(docid, new HashMap<>());
                if (oneTermFreq.size() == 0) {
                    tokenOnCollection.putIfAbsent(docid, oneTermFreq);
                }
                oneTermFreq.put(token, docFreq);
            }
        }
        return tokenOnCollection;
    }

}
