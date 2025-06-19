import React from "react";
import { View, Text, FlatList, TouchableOpacity } from "react-native";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { apiClient } from "@/lib/api-client";
import { Loading } from "./Loading";
import { LeagueLeader } from "@nba/api/types";

export function LeagueLeadersCard() {
  const router = useRouter();
  const currentSeasonYear = new Date().getFullYear();

  const { data: leaders, isLoading } = useQuery({
    queryKey: ["analytics", "league-leaders", currentSeasonYear],
    queryFn: () => apiClient.getLeagueLeaders(currentSeasonYear - 1, 5),
  });

  const handlePlayerPress = (leader: LeagueLeader) => {
    router.push(`/player/${leader.player.person_id}`);
  };

  return (
    <View className="bg-white rounded-lg shadow-sm p-4">
      <Text className="text-lg font-bold text-gray-900 mb-4">
        Scoring Leaders
      </Text>

      {isLoading ? (
        <Loading text="Loading leaders..." />
      ) : !leaders || leaders.length === 0 ? (
        <Text className="text-gray-500 text-center py-8">
          No data available
        </Text>
      ) : (
        <FlatList
          data={leaders}
          keyExtractor={(item) => item.player.person_id.toString()}
          renderItem={({ item, index }) => (
            <TouchableOpacity
              onPress={() => handlePlayerPress(item)}
              className="flex-row items-center py-3 border-b border-gray-100 last:border-b-0"
            >
              <Text className="text-lg font-bold text-gray-400 w-8">
                {index + 1}
              </Text>
              <View className="flex-1 ml-3">
                <Text className="font-semibold text-gray-900">
                  {item.player.full_name}
                </Text>
                <Text className="text-sm text-gray-500">
                  {item.team.team_abbrev}
                </Text>
              </View>
              <Text className="text-lg font-bold text-blue-600">
                {item.avg_points.toFixed(1)}
              </Text>
            </TouchableOpacity>
          )}
          showsVerticalScrollIndicator={false}
          scrollEnabled={false}
        />
      )}
    </View>
  );
}