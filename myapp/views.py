from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from rapidfuzz.distance.JaroWinkler import normalized_similarity
from itertools import permutations
import re

def read_file(file) -> pd.DataFrame:
    try:
        filename = file.name
        if filename.endswith('.csv'):
            return pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type. Only .csv and .xlsx are allowed.")
    except Exception as e:
        raise ValueError(f"Failed to read {filename}: {str(e)}")

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_files(request):
    try:
        master_file = request.FILES.get('master_file')
        buyer_file = request.FILES.get('buyer_file')

        if not master_file or not buyer_file:
            return Response({"detail": "Both master_file and buyer_file are required."},
                            status=status.HTTP_400_BAD_REQUEST)

        master_df = read_file(master_file)
        buyer_df = read_file(buyer_file)

        # Validate required columns
        if 'Master_Code' not in master_df.columns or 'Master_Name' not in master_df.columns:
            return Response({"detail": "Master dataset must contain 'Master_Code' and 'Master_Name' columns."},
                            status=status.HTTP_400_BAD_REQUEST)
        if 'Buyer_Name' not in buyer_df.columns:
            return Response({"detail": "Test dataset must contain 'Buyer_Name' column."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Normalize names
        master_df['Master_Name_Clean'] = master_df['Master_Name'].str.lower().str.strip()
        buyer_df['Buyer_Name_Clean'] = buyer_df['Buyer_Name'].str.lower().str.strip()

        similar_match = []

        def clean_company_name_for_jaccard(name):
            suffixes = r"\b(incorporated|inc|llc|ltd|limited|corp|corporation|plc|co|company|pvt|private)\b"
            name = name.lower()
            name = re.sub(suffixes, '', name)
            name = re.sub(r'\s+', ' ', name)
            return name.strip()

        def permuted_winkler_distance(a, b):
            tokens = a.split()
            max_sim = 0.0
            for perm in permutations(tokens):
                permuted = " ".join(perm)
                sim = normalized_similarity(permuted, b) / 100
                if sim > max_sim:
                    max_sim = sim
            return 1.0 - max_sim

        def jaccard_distance(a, b):
            a_clean = clean_company_name_for_jaccard(a)
            b_clean = clean_company_name_for_jaccard(b)
            set_a = set(a_clean.split())
            set_b = set(b_clean.split())
            intersection = set_a & set_b
            union = set_a | set_b
            if not union:
                return 1.0
            return 1.0 - len(intersection) / len(union)

        for i, test_name_clean in enumerate(buyer_df['Buyer_Name_Clean']):
            buyer_name = buyer_df['Buyer_Name'].iloc[i]

            exact_match = master_df[master_df['Master_Name_Clean'] == test_name_clean]
            if not exact_match.empty:
                continue

            winkler_distances = master_df['Master_Name_Clean'].apply(
                lambda master_clean: permuted_winkler_distance(test_name_clean, master_clean)
            )
            jaccard_distances = master_df['Master_Name_Clean'].apply(
                lambda master_clean: jaccard_distance(test_name_clean, master_clean)
            )

            top_winkler = winkler_distances.nsmallest(10).index.tolist()
            top_jaccard = jaccard_distances.nsmallest(10).index.tolist()

            interleaved = []
            for w, j in zip(top_winkler, top_jaccard):
                interleaved.append(w)
                interleaved.append(j)

            unique_indices = list(dict.fromkeys(interleaved))

            top_matches = master_df.loc[unique_indices[:10], 'Master_Name'].tolist()

            similar_match.append({
                "Buyer_Name": buyer_name,
                "Top_Matches": top_matches
            })

        return Response({"matches": similar_match}, status=status.HTTP_200_OK)

    except ValueError as ve:
        return Response({"detail": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"detail": f"Error processing files: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

