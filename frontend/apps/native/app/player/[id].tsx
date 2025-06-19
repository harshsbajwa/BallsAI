import { PlayerStats } from "@nba/api/types";
import { useQuery } from "@tanstack/react-query";
import { useLocalSearchParams } from "expo-router";
import React from "react";
import { View, Text, ScrollView } from "react-native";

import { Loading } from "@/components/Loading";

import { apiClient } from "@/lib/api-client";

export default function PlayerDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const playerId = parseInt(id, 10);

  const { data: player, isLoading: playerLoading } = useQuery({
    queryKey: ["player", playerId],
    queryFn: () => apiClient.getPlayer(playerId),
    enabled: !!playerId,
  });

  const { data: playerStats, isLoading: statsLoading } = useQuery({
    queryKey: ["player", playerId, "stats"],
    queryFn: () => apiClient.getPlayerStats(playerId, 10),
    enabled: !!playerId,
  });

  if (playerLoading) {
    return (
      <View className="flex-1 bg-gray-50">
        <Loading text="Loading player..." />
      </View>
    );
  }

  if (!player) {
    return (
      <View className="flex-1 bg-gray-50 items-center justify-center">
        <Text className="text-red-500 text-lg">Player not found</Text>
      </View>
    );
  }

  return (
    <ScrollView className="flex-1 bg-gray-50">
      {/* Player Header */}
      <View className="bg-white p-6 shadow-sm">
        <Text className="text-3xl font-bold text-gray-900 mb-2">
          {player.full_name}
        </Text>
        <View className="flex-row items-center space-x-4">
          <Text className="text-lg text-gray-600">
            {player.primary_position}
          </Text>
          {player.draft_year && (
            <Text className="text-lg text-gray-600">
              Draft {player.draft_year}
            </Text>
          )}
        </View>
      </View>

      {/* Player Info */}
      <View className="bg-white mx-4 mt-4 rounded-lg shadow-sm p-4">
        <Text className="text-lg font-bold text-gray-900 mb-4">
          Player Information
        </Text>
        <View className="space-y-3">
          {player.height && player.body_weight && (
            <View className="flex-row justify-between">
              <Text className="text-gray-600">Height & Weight:</Text>
              <Text className="font-semibold">
                {player.height}" • {player.body_weight} lbs
              </Text>
            </View>
          )}
          {player.birthdate && (
            <View className="flex-row justify-between">
              <Text className="text-gray-600">Born:</Text>
              <Text className="font-semibold">
                {new Date(player.birthdate).toLocaleDateString()}
              </Text>
            </View>
          )}
          {player.country && (
            <View className="flex-row justify-between">
              <Text className="text-gray-600">Country:</Text>
              <Text className="font-semibold">{player.country}</Text>
            </View>
          )}
          {player.draft_year && player.draft_round && player.draft_number && (
            <View className="flex-row justify-between">
              <Text className="text-gray-600">Draft Position:</Text>
              <Text className="font-semibold">
                {player.draft_year} - Round {player.draft_round}, Pick{" "}
                {player.draft_number}
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Recent Stats */}
      <View className="bg-white mx-4 my-4 rounded-lg shadow-sm p-4">
        <Text className="text-lg font-bold text-gray-900 mb-4">
          Recent Games
        </Text>
        {statsLoading ? (
          <Loading text="Loading stats..." />
        ) : playerStats?.items.length === 0 ? (
          <Text className="text-gray-500 text-center py-4">
            No recent games available
          </Text>
        ) : (
          <View className="space-y-3">
            {playerStats?.items.slice(0, 5).map((stat: PlayerStats) => (
              <View
                key={stat.game_id}
                className="border-b border-gray-100 pb-3 last:border-b-0"
              >
                <View className="flex-row justify-between items-center mb-2">
                  <Text className="text-sm text-gray-500">
                    {new Date(stat.game_date).toLocaleDateString()}
                  </Text>
                  <Text className="font-semibold text-lg">{stat.points} PTS</Text>
                </View>
                <View className="flex-row justify-around text-sm">
                  <Text className="text-gray-600">{stat.rebounds_total} REB</Text>
                  <Text className="text-gray-600">{stat.assists} AST</Text>
                  <Text className="text-gray-600">{stat.steals} STL</Text>
                  <Text className="text-gray-600">{stat.blocks} BLK</Text>
                </View>
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
}