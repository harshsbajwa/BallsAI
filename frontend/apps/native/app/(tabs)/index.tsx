import { View, Text, ScrollView, RefreshControl } from "react-native";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { LiveScoresCard } from "@/components/LiveScoresCard";
import { PlayerSearchCard } from "@/components/PlayerSearchCard";
import { LeagueLeadersCard } from "@/components/LeagueLeadersCard";
import { GamePredictionsCard } from "@/components/GamePredictionsCard";
import { apiClient } from "@/lib/api-client";

export default function HomeScreen() {
  const [refreshing, setRefreshing] = useState(false);

  const {
    data: todaysGames,
    refetch,
    isLoading,
  } = useQuery({
    queryKey: ["games", "today"],
    queryFn: () => apiClient.getTodaysGames(),
  });

  const onRefresh = async () => {
    setRefreshing(true);
    await refetch();
    setRefreshing(false);
  };

  return (
    <ScrollView
      className="flex-1 bg-gray-50"
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Header */}
      <View className="bg-blue-600 px-4 py-8 pt-16">
        <Text className="text-white text-3xl font-bold mb-2">BallsAI</Text>
        <Text className="text-blue-100 text-lg">
          Real-time NBA stats and predictions
        </Text>
      </View>

      {/* Content */}
      <View className="p-4 space-y-6">
        <PlayerSearchCard />
        <LiveScoresCard games={todaysGames} loading={isLoading} />
        <GamePredictionsCard />
        <LeagueLeadersCard />
      </View>
    </ScrollView>
  );
}